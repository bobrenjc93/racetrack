"""Fused SwiGLU activation for CuteDSL backend.

Inline CUDA C++ kernel: silu(gate) * up in a single pass,
reading bf16 directly and writing bf16 output.
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

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float(float v) { return v; }
template <> __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_float(__nv_half v) { return __half2float(v); }

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __nv_half from_float(float v) { return __float2half(v); }

template <typename T>
__global__ void swiglu_kernel(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ out,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    float g = to_float(gate[idx]);
    float u = to_float(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    out[idx] = from_float<T>(silu_g * u);
}

torch::Tensor swiglu_cuda(torch::Tensor gate, torch::Tensor up) {
    auto g_c = gate.contiguous();
    auto u_c = up.contiguous();
    int64_t n = g_c.numel();
    auto out = torch::empty_like(g_c);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    if (g_c.dtype() == torch::kBFloat16) {
        swiglu_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)g_c.data_ptr(),
            (const __nv_bfloat16*)u_c.data_ptr(),
            (__nv_bfloat16*)out.data_ptr(), n);
    } else if (g_c.dtype() == torch::kFloat16) {
        swiglu_kernel<__nv_half><<<blocks, threads, 0, stream>>>(
            (const __nv_half*)g_c.data_ptr(),
            (const __nv_half*)u_c.data_ptr(),
            (__nv_half*)out.data_ptr(), n);
    } else {
        swiglu_kernel<float><<<blocks, threads, 0, stream>>>(
            g_c.data_ptr<float>(),
            u_c.data_ptr<float>(),
            out.data_ptr<float>(), n);
    }
    return out;
}
"""

    cpp_src = r"""
torch::Tensor swiglu_cuda(torch::Tensor gate, torch::Tensor up);
"""

    _cuda_module = load_inline(
        name="cutedsl_swiglu",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=["swiglu_cuda"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return _cuda_module


def fused_swiglu(gate, up, *, fallback):
    del fallback
    mod = _get_cuda_module()
    return mod.swiglu_cuda(gate, up)
