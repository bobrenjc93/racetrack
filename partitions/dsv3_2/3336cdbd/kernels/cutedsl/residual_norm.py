"""Fused residual add + RMS norm for CuteDSL backend.

Inline CUDA C++ kernel: one thread block per row computes
residual + update → hidden, then RMS norm → normed. Single pass,
warp-level reduction for variance, no intermediate tensors.
"""
from __future__ import annotations

import torch

try:
    import cutlass  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False

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
#include <float.h>

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float(float v) { return v; }
template <> __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_float(__nv_half v) { return __half2float(v); }

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __nv_half from_float(float v) { return __float2half(v); }

// Each thread block processes one row. blockDim.x = next_pow2(cols), capped at 1024.
// For cols > 1024, each thread loops over multiple elements.
template <typename T>
__global__ void residual_norm_kernel(
    const T* __restrict__ update,
    const T* __restrict__ residual,
    const float* __restrict__ weight,
    T* __restrict__ out_hidden,
    T* __restrict__ out_normed,
    int cols,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Each thread accumulates partial sum of squares over its elements
    float sum_sq = 0.0f;
    for (int c = tid; c < cols; c += nthreads) {
        float u = to_float(update[row * cols + c]);
        float r = to_float(residual[row * cols + c]);
        float h = u + r;
        out_hidden[row * cols + c] = from_float<T>(h);
        sum_sq += h * h;
    }

    // Warp-level reduction of sum_sq
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    // Cross-warp reduction via shared memory (up to 32 warps for 1024 threads)
    __shared__ float shared[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int n_warps = (nthreads + 31) / 32;
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < n_warps; i++) total += shared[i];
        shared[0] = total;
    }
    __syncthreads();

    float variance = shared[0] / (float)cols;
    float scale = rsqrtf(variance + eps);

    for (int c = tid; c < cols; c += nthreads) {
        float u = to_float(update[row * cols + c]);
        float r = to_float(residual[row * cols + c]);
        float h = u + r;
        float w = weight[c];
        out_normed[row * cols + c] = from_float<T>(h * scale * w);
    }
}

std::tuple<torch::Tensor, torch::Tensor> residual_norm_cuda(
    torch::Tensor update, torch::Tensor residual, torch::Tensor weight,
    double eps
) {
    auto u_c = update.contiguous();
    auto r_c = residual.contiguous();
    auto w_c = weight.contiguous();
    int64_t cols = u_c.size(-1);
    int64_t n_rows = u_c.numel() / cols;

    auto out_hidden = torch::empty_like(u_c);
    auto out_normed = torch::empty_like(u_c);

    // Use up to 1024 threads, round up to next multiple of 32
    int nthreads = ((int)cols + 31) / 32 * 32;
    if (nthreads > 1024) nthreads = 1024;

    auto stream = c10::cuda::getCurrentCUDAStream();
    if (u_c.dtype() == torch::kBFloat16) {
        residual_norm_kernel<__nv_bfloat16><<<n_rows, nthreads, 0, stream>>>(
            (const __nv_bfloat16*)u_c.data_ptr(),
            (const __nv_bfloat16*)r_c.data_ptr(),
            w_c.data_ptr<float>(),
            (__nv_bfloat16*)out_hidden.data_ptr(),
            (__nv_bfloat16*)out_normed.data_ptr(),
            cols, (float)eps);
    } else if (u_c.dtype() == torch::kFloat16) {
        residual_norm_kernel<__nv_half><<<n_rows, nthreads, 0, stream>>>(
            (const __nv_half*)u_c.data_ptr(),
            (const __nv_half*)r_c.data_ptr(),
            w_c.data_ptr<float>(),
            (__nv_half*)out_hidden.data_ptr(),
            (__nv_half*)out_normed.data_ptr(),
            cols, (float)eps);
    } else {
        residual_norm_kernel<float><<<n_rows, nthreads, 0, stream>>>(
            u_c.data_ptr<float>(),
            r_c.data_ptr<float>(),
            w_c.data_ptr<float>(),
            out_hidden.data_ptr<float>(),
            out_normed.data_ptr<float>(),
            cols, (float)eps);
    }
    return {out_normed, out_hidden};
}
"""

    cpp_src = r"""
std::tuple<torch::Tensor, torch::Tensor> residual_norm_cuda(
    torch::Tensor update, torch::Tensor residual, torch::Tensor weight,
    double eps);
"""

    _cuda_module = load_inline(
        name="cutedsl_residual_norm",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=["residual_norm_cuda"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return _cuda_module


def fused_residual_norm(
    update: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float,
    fallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    del fallback
    mod = _get_cuda_module()
    shape = update.shape
    normed, hidden = mod.residual_norm_cuda(update, residual, norm_weight, eps)
    return normed.view(shape), hidden.view(shape)
