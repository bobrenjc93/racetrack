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
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float(float v) { return v; }
template <> __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_float(__nv_half v) { return __half2float(v); }

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __nv_half from_float(float v) { return __float2half(v); }

template <typename T>
__global__ void rms_norm_kernel(
    const T* __restrict__ x,
    const float* __restrict__ weight,
    T* __restrict__ out,
    int cols,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float sum_sq = 0.0f;
    for (int c = tid; c < cols; c += nthreads) {
        float v = to_float(x[row * cols + c]);
        sum_sq += v * v;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);

    __shared__ float shared[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int n_warps = (nthreads + 31) / 32;
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < n_warps; i++) total += shared[i];
        shared[0] = rsqrtf(total / (float)cols + eps);
    }
    __syncthreads();
    float scale = shared[0];

    for (int c = tid; c < cols; c += nthreads) {
        float v = to_float(x[row * cols + c]);
        out[row * cols + c] = from_float<T>(v * scale * weight[c]);
    }
}

template <typename T>
__global__ void rope_kernel(
    const T* __restrict__ x,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    T* __restrict__ out,
    int half_d
) {
    int row = blockIdx.x;
    int pair = threadIdx.x;
    if (pair >= half_d) return;

    int idx_even = row * half_d * 2 + pair * 2;
    int idx_odd = idx_even + 1;
    int freq_idx = row * half_d + pair;

    float x0 = to_float(x[idx_even]);
    float x1 = to_float(x[idx_odd]);
    float c = cos_cache[freq_idx];
    float s = sin_cache[freq_idx];

    out[idx_even] = from_float<T>(x0 * c - x1 * s);
    out[idx_odd] = from_float<T>(x0 * s + x1 * c);
}

void rms_norm_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor out,
    int64_t n_rows, int64_t cols, double eps
) {
    int nthreads = ((int)cols + 31) / 32 * 32;
    if (nthreads > 1024) nthreads = 1024;
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (x.dtype() == torch::kBFloat16) {
        rms_norm_kernel<__nv_bfloat16><<<n_rows, nthreads, 0, stream>>>(
            (const __nv_bfloat16*)x.data_ptr(),
            weight.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
            cols, (float)eps);
    } else if (x.dtype() == torch::kFloat16) {
        rms_norm_kernel<__nv_half><<<n_rows, nthreads, 0, stream>>>(
            (const __nv_half*)x.data_ptr(),
            weight.data_ptr<float>(), (__nv_half*)out.data_ptr(),
            cols, (float)eps);
    } else {
        rms_norm_kernel<float><<<n_rows, nthreads, 0, stream>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(), out.data_ptr<float>(),
            cols, (float)eps);
    }
}

void rope_apply_cuda(
    torch::Tensor x, torch::Tensor cos_cache, torch::Tensor sin_cache,
    torch::Tensor out, int64_t n_rows, int64_t half_d
) {
    int threads = (int)half_d;
    if (threads > 1024) threads = 1024;
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (x.dtype() == torch::kBFloat16) {
        rope_kernel<__nv_bfloat16><<<n_rows, threads, 0, stream>>>(
            (const __nv_bfloat16*)x.data_ptr(),
            cos_cache.data_ptr<float>(), sin_cache.data_ptr<float>(),
            (__nv_bfloat16*)out.data_ptr(), half_d);
    } else if (x.dtype() == torch::kFloat16) {
        rope_kernel<__nv_half><<<n_rows, threads, 0, stream>>>(
            (const __nv_half*)x.data_ptr(),
            cos_cache.data_ptr<float>(), sin_cache.data_ptr<float>(),
            (__nv_half*)out.data_ptr(), half_d);
    } else {
        rope_kernel<float><<<n_rows, threads, 0, stream>>>(
            x.data_ptr<float>(),
            cos_cache.data_ptr<float>(), sin_cache.data_ptr<float>(),
            out.data_ptr<float>(), half_d);
    }
}
"""

    cpp_src = r"""
void rms_norm_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor out,
    int64_t n_rows, int64_t cols, double eps);
void rope_apply_cuda(
    torch::Tensor x, torch::Tensor cos_cache, torch::Tensor sin_cache,
    torch::Tensor out, int64_t n_rows, int64_t half_d);
"""

    _cuda_module = load_inline(
        name="cutedsl_norm_rope",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=["rms_norm_cuda", "rope_apply_cuda"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return _cuda_module


def _rope_cache(positions, rotary_dim, *, base, dtype):
    device = positions.device
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) / max(half, 1))
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def fused_norm_rope(
    q_c, q_weight, kv_c, kv_weight, k_pe, positions,
    *, eps, rope_base, fallback,
):
    del fallback
    mod = _get_cuda_module()

    q_c_cont = q_c.contiguous()
    q_out = torch.empty_like(q_c_cont)
    mod.rms_norm_cuda(
        q_c_cont, q_weight.contiguous().float(), q_out,
        q_c_cont.shape[0], q_c_cont.shape[-1], eps,
    )

    kv_c_cont = kv_c.contiguous()
    kv_out = torch.empty_like(kv_c_cont)
    mod.rms_norm_cuda(
        kv_c_cont, kv_weight.contiguous().float(), kv_out,
        kv_c_cont.shape[0], kv_c_cont.shape[-1], eps,
    )

    k_pe_cont = k_pe.contiguous()
    rotary_dim = k_pe_cont.shape[-1]
    half_d = rotary_dim // 2
    cos, sin = _rope_cache(positions, rotary_dim, base=rope_base, dtype=torch.float32)
    cos = cos.contiguous()
    sin = sin.contiguous()
    k_pe_out = torch.empty_like(k_pe_cont)
    mod.rope_apply_cuda(k_pe_cont, cos, sin, k_pe_out, k_pe_cont.shape[0], half_d)

    return q_out, kv_out, k_pe_out
