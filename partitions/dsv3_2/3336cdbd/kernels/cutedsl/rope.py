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
__global__ void rope_kernel(
    const T* __restrict__ x,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    T* __restrict__ out,
    int half_d,
    int freq_stride
) {
    int row = blockIdx.x;
    int pair = threadIdx.x;
    if (pair >= half_d) return;

    int idx_even = row * half_d * 2 + pair * 2;
    int idx_odd = idx_even + 1;
    int freq_idx = (row % freq_stride) * half_d + pair;

    float x0 = to_float(x[idx_even]);
    float x1 = to_float(x[idx_odd]);
    float c = cos_cache[freq_idx];
    float s = sin_cache[freq_idx];

    out[idx_even] = from_float<T>(x0 * c - x1 * s);
    out[idx_odd] = from_float<T>(x0 * s + x1 * c);
}

torch::Tensor rope_cuda(
    torch::Tensor x, torch::Tensor cos_cache, torch::Tensor sin_cache,
    int64_t n_rows, int64_t half_d, int64_t freq_rows
) {
    auto x_c = x.contiguous();
    auto out = torch::empty_like(x_c);
    int threads = (int)half_d;
    if (threads > 1024) threads = 1024;
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (x_c.dtype() == torch::kBFloat16) {
        rope_kernel<__nv_bfloat16><<<n_rows, threads, 0, stream>>>(
            (const __nv_bfloat16*)x_c.data_ptr(),
            cos_cache.data_ptr<float>(), sin_cache.data_ptr<float>(),
            (__nv_bfloat16*)out.data_ptr(), half_d, freq_rows);
    } else if (x_c.dtype() == torch::kFloat16) {
        rope_kernel<__nv_half><<<n_rows, threads, 0, stream>>>(
            (const __nv_half*)x_c.data_ptr(),
            cos_cache.data_ptr<float>(), sin_cache.data_ptr<float>(),
            (__nv_half*)out.data_ptr(), half_d, freq_rows);
    } else {
        rope_kernel<float><<<n_rows, threads, 0, stream>>>(
            x_c.data_ptr<float>(),
            cos_cache.data_ptr<float>(), sin_cache.data_ptr<float>(),
            out.data_ptr<float>(), half_d, freq_rows);
    }
    return out;
}
"""

    cpp_src = r"""
torch::Tensor rope_cuda(
    torch::Tensor x, torch::Tensor cos_cache, torch::Tensor sin_cache,
    int64_t n_rows, int64_t half_d, int64_t freq_rows);
"""

    _cuda_module = load_inline(
        name="cutedsl_rope",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=["rope_cuda"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return _cuda_module


def fused_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    fallback,
) -> torch.Tensor:
    del fallback
    shape = x.shape
    d = shape[-1]
    half_d = d // 2

    freqs_flat = freqs_cis.view(-1, half_d)
    cos_cache = freqs_flat.real.contiguous().to(torch.float32)
    sin_cache = freqs_flat.imag.contiguous().to(torch.float32)

    x_flat = x.contiguous().view(-1, d)
    n_rows = x_flat.shape[0]
    freq_rows = cos_cache.shape[0]

    mod = _get_cuda_module()
    out = mod.rope_cuda(x_flat, cos_cache, sin_cache, n_rows, half_d, freq_rows)
    return out.view(shape)
