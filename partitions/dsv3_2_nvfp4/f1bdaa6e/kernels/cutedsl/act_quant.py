"""FP8 per-block activation quantization for CuteDSL backend.

Uses inline CUDA C++ kernel for single-pass fusion: one kernel does
abs → amax (warp reduction) → scale → clamp → fp8 cast. This is the
standard CUTLASS approach — CUTLASS itself is C++ CUDA code.
"""
from __future__ import annotations

import torch

try:
    import cutlass  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False

QUANT_BLOCK = 128

_cuda_module = None


def _get_cuda_module():
    global _cuda_module
    if _cuda_module is not None:
        return _cuda_module

    from torch.utils.cpp_extension import load_inline

    cuda_src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <float.h>

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float(float v) { return v; }
template <> __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_float(__nv_half v) { return __half2float(v); }

template <typename T>
__global__ void act_quant_kernel(
    const T* __restrict__ x,
    __nv_fp8_e4m3* __restrict__ out_fp8,
    float* __restrict__ out_scale,
    int n_elements,
    int n_cols,
    int n_qb_per_row
) {
    int row = blockIdx.x;
    int qb = blockIdx.y;
    int tid = threadIdx.x;
    int col = qb * 128 + tid;

    float val = (col < n_cols) ? to_float(x[row * n_cols + col]) : 0.0f;
    float abs_val = fabsf(val);

    float warp_max = abs_val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xffffffff, warp_max, offset));
    }

    __shared__ float shared_max[4];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared_max[warp_id] = warp_max;
    __syncthreads();

    float block_max;
    if (tid < 4) {
        block_max = shared_max[tid];
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(0xf, block_max, offset));
        }
        if (tid == 0) shared_max[0] = fmaxf(block_max, 1e-4f);
    }
    __syncthreads();

    float amax = shared_max[0];
    float scale = amax / 448.0f;
    float scaled = val / scale;
    float clamped = fminf(fmaxf(scaled, -448.0f), 448.0f);

    if (col < n_cols) {
        out_fp8[row * n_cols + col] = __nv_fp8_e4m3(clamped);
    }
    if (tid == 0) {
        out_scale[row * n_qb_per_row + qb] = scale;
    }
}

template <typename T>
__global__ void swiglu_quant_kernel(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    __nv_fp8_e4m3* __restrict__ out_fp8,
    float* __restrict__ out_scale,
    int n_cols,
    int n_qb_per_row
) {
    int row = blockIdx.x;
    int qb = blockIdx.y;
    int tid = threadIdx.x;
    int col = qb * 128 + tid;

    float g = (col < n_cols) ? to_float(gate[row * n_cols + col]) : 0.0f;
    float u = (col < n_cols) ? to_float(up[row * n_cols + col]) : 0.0f;

    float silu_g = g / (1.0f + expf(-g));
    float val = silu_g * u;
    float abs_val = fabsf(val);

    float warp_max = abs_val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xffffffff, warp_max, offset));
    }

    __shared__ float shared_max[4];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared_max[warp_id] = warp_max;
    __syncthreads();

    float block_max;
    if (tid < 4) {
        block_max = shared_max[tid];
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(0xf, block_max, offset));
        }
        if (tid == 0) shared_max[0] = fmaxf(block_max, 1e-4f);
    }
    __syncthreads();

    float amax = shared_max[0];
    float scale = amax / 448.0f;
    float scaled = val / scale;
    float clamped = fminf(fmaxf(scaled, -448.0f), 448.0f);

    if (col < n_cols) {
        out_fp8[row * n_cols + col] = __nv_fp8_e4m3(clamped);
    }
    if (tid == 0) {
        out_scale[row * n_qb_per_row + qb] = scale;
    }
}

#define LAUNCH_ACT_QUANT(T, x_c) do {                                            \
    act_quant_kernel<T><<<grid, 128, 0, stream>>>(                               \
        (const T*)x_c.data_ptr(),                                                \
        (__nv_fp8_e4m3*)out_fp8.data_ptr(),                                      \
        out_scale.data_ptr<float>(),                                             \
        n_cols, n_cols, n_qb);                                                   \
} while(0)

std::tuple<torch::Tensor, torch::Tensor> act_quant_cuda(
    torch::Tensor x, int64_t n_rows, int64_t n_cols
) {
    auto x_c = x.contiguous();
    int n_qb = (n_cols + 127) / 128;
    dim3 grid(n_rows, n_qb);
    auto out_fp8 = torch::empty_like(x_c, torch::TensorOptions()
        .dtype(torch::kFloat8_e4m3fn).device(x.device()));
    auto out_scale = torch::empty({n_rows, n_qb},
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    auto stream = c10::cuda::getCurrentCUDAStream();
    if (x_c.dtype() == torch::kBFloat16) { LAUNCH_ACT_QUANT(__nv_bfloat16, x_c); }
    else if (x_c.dtype() == torch::kFloat16) { LAUNCH_ACT_QUANT(__nv_half, x_c); }
    else { LAUNCH_ACT_QUANT(float, x_c); }
    return {out_fp8, out_scale};
}

#define LAUNCH_SWIGLU_QUANT(T, g_c, u_c) do {                                   \
    swiglu_quant_kernel<T><<<grid, 128, 0, stream>>>(                            \
        (const T*)g_c.data_ptr(),                                                \
        (const T*)u_c.data_ptr(),                                                \
        (__nv_fp8_e4m3*)out_fp8.data_ptr(),                                      \
        out_scale.data_ptr<float>(),                                             \
        n_cols, n_qb);                                                           \
} while(0)

std::tuple<torch::Tensor, torch::Tensor> swiglu_quant_cuda(
    torch::Tensor gate, torch::Tensor up, int64_t n_rows, int64_t n_cols
) {
    auto g_c = gate.contiguous();
    auto u_c = up.contiguous();
    int n_qb = (n_cols + 127) / 128;
    dim3 grid(n_rows, n_qb);
    auto out_fp8 = torch::empty_like(g_c, torch::TensorOptions()
        .dtype(torch::kFloat8_e4m3fn).device(gate.device()));
    auto out_scale = torch::empty({n_rows, n_qb},
        torch::TensorOptions().dtype(torch::kFloat32).device(gate.device()));

    auto stream = c10::cuda::getCurrentCUDAStream();
    if (g_c.dtype() == torch::kBFloat16) { LAUNCH_SWIGLU_QUANT(__nv_bfloat16, g_c, u_c); }
    else if (g_c.dtype() == torch::kFloat16) { LAUNCH_SWIGLU_QUANT(__nv_half, g_c, u_c); }
    else { LAUNCH_SWIGLU_QUANT(float, g_c, u_c); }
    return {out_fp8, out_scale};
}
"""

    cpp_src = r"""
std::tuple<torch::Tensor, torch::Tensor> act_quant_cuda(
    torch::Tensor x, int64_t n_rows, int64_t n_cols);
std::tuple<torch::Tensor, torch::Tensor> swiglu_quant_cuda(
    torch::Tensor gate, torch::Tensor up, int64_t n_rows, int64_t n_cols);
"""

    _cuda_module = load_inline(
        name="cutedsl_act_quant",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=["act_quant_cuda", "swiglu_quant_cuda"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return _cuda_module


def fused_act_quant(x, *, fallback):
    del fallback
    mod = _get_cuda_module()
    shape = x.shape
    N = shape[-1]
    n_rows = x.numel() // N
    n_groups = (N + QUANT_BLOCK - 1) // QUANT_BLOCK

    fp8, scale = mod.act_quant_cuda(x, n_rows, N)
    return fp8, scale.view(*shape[:-1], n_groups)


def fused_swiglu_quant(gate, up, *, fallback):
    del fallback
    mod = _get_cuda_module()
    shape = gate.shape
    N = shape[-1]
    n_rows = gate.numel() // N
    n_groups = (N + QUANT_BLOCK - 1) // QUANT_BLOCK

    fp8, scale = mod.swiglu_quant_cuda(gate, up, n_rows, N)
    return fp8, scale.view(*shape[:-1], n_groups)
