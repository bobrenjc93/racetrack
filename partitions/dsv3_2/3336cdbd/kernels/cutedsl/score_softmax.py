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
#include <float.h>
#include <math.h>

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float(float v) { return v; }
template <> __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_float(__nv_half v) { return __half2float(v); }

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __nv_half from_float(float v) { return __float2half(v); }

// Each block handles one (batch_seq, head) pair.
// Three-pass online softmax over the time dimension:
//   1. Find global max
//   2. Sum exp(score - max)
//   3. Write softmax probs
template <typename T>
__global__ void score_softmax_kernel(
    const T* __restrict__ nope,
    const T* __restrict__ rope,
    const float* __restrict__ mask,
    T* __restrict__ out,
    float softmax_scale,
    int T_dim,
    int H,
    int stride_score_bh,
    int stride_mask_b
) {
    int bs = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    int score_base = bs * H * T_dim + h * T_dim;
    int mask_base = bs * T_dim;

    // Pass 1: find max
    float local_max = -FLT_MAX;
    for (int t = tid; t < T_dim; t += nthreads) {
        float n = to_float(nope[score_base + t]);
        float r = to_float(rope[score_base + t]);
        float m = mask[mask_base + t];
        float s = (n + r) * softmax_scale + m;
        if (s > local_max) local_max = s;
    }

    // Warp reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));

    __shared__ float shared[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int n_warps = (nthreads + 31) / 32;
    if (lane_id == 0) shared[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float gmax = -FLT_MAX;
        for (int i = 0; i < n_warps; i++) gmax = fmaxf(gmax, shared[i]);
        shared[0] = gmax;
    }
    __syncthreads();
    float global_max = shared[0];

    // Pass 2: sum of exp
    float local_sum = 0.0f;
    for (int t = tid; t < T_dim; t += nthreads) {
        float n = to_float(nope[score_base + t]);
        float r = to_float(rope[score_base + t]);
        float m = mask[mask_base + t];
        float s = (n + r) * softmax_scale + m;
        local_sum += expf(s - global_max);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    if (lane_id == 0) shared[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < n_warps; i++) total += shared[i];
        shared[0] = total;
    }
    __syncthreads();
    float global_sum = shared[0];

    // Pass 3: write normalized probs
    int out_base = bs * H * T_dim + h * T_dim;
    for (int t = tid; t < T_dim; t += nthreads) {
        float n = to_float(nope[score_base + t]);
        float r = to_float(rope[score_base + t]);
        float m = mask[mask_base + t];
        float s = (n + r) * softmax_scale + m;
        float prob = expf(s - global_max) / global_sum;
        out[out_base + t] = from_float<T>(prob);
    }
}

torch::Tensor score_softmax_cuda(
    torch::Tensor nope, torch::Tensor rope, torch::Tensor mask,
    double softmax_scale, int64_t BS, int64_t H, int64_t T
) {
    auto nope_c = nope.contiguous();
    auto rope_c = rope.contiguous();
    auto mask_c = mask.contiguous().to(torch::kFloat32);
    auto out = torch::empty_like(nope_c);

    int nthreads = ((int)T + 31) / 32 * 32;
    if (nthreads > 1024) nthreads = 1024;
    dim3 grid(BS, H);
    auto stream = c10::cuda::getCurrentCUDAStream();

    if (nope_c.dtype() == torch::kBFloat16) {
        score_softmax_kernel<__nv_bfloat16><<<grid, nthreads, 0, stream>>>(
            (const __nv_bfloat16*)nope_c.data_ptr(),
            (const __nv_bfloat16*)rope_c.data_ptr(),
            mask_c.data_ptr<float>(),
            (__nv_bfloat16*)out.data_ptr(),
            (float)softmax_scale, T, H, H*T, T);
    } else if (nope_c.dtype() == torch::kFloat16) {
        score_softmax_kernel<__nv_half><<<grid, nthreads, 0, stream>>>(
            (const __nv_half*)nope_c.data_ptr(),
            (const __nv_half*)rope_c.data_ptr(),
            mask_c.data_ptr<float>(),
            (__nv_half*)out.data_ptr(),
            (float)softmax_scale, T, H, H*T, T);
    } else {
        score_softmax_kernel<float><<<grid, nthreads, 0, stream>>>(
            nope_c.data_ptr<float>(),
            rope_c.data_ptr<float>(),
            mask_c.data_ptr<float>(),
            out.data_ptr<float>(),
            (float)softmax_scale, T, H, H*T, T);
    }
    return out;
}
"""

    cpp_src = r"""
torch::Tensor score_softmax_cuda(
    torch::Tensor nope, torch::Tensor rope, torch::Tensor mask,
    double softmax_scale, int64_t BS, int64_t H, int64_t T);
"""

    _cuda_module = load_inline(
        name="cutedsl_score_softmax",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=["score_softmax_cuda"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return _cuda_module


def fused_score_softmax(
    scores_nope: torch.Tensor,
    scores_rope: torch.Tensor,
    index_mask: torch.Tensor,
    *,
    softmax_scale: float,
    fallback,
) -> torch.Tensor:
    del fallback
    orig_shape = scores_nope.shape

    if scores_nope.dim() == 4:
        B, S, H, T = scores_nope.shape
    elif scores_nope.dim() == 3:
        H, T = scores_nope.shape[-2], scores_nope.shape[-1]
        B = scores_nope.shape[0]
        S = 1
        scores_nope = scores_nope.unsqueeze(1)
        scores_rope = scores_rope.unsqueeze(1)
    else:
        raise ValueError(f"Expected 3D or 4D scores, got {scores_nope.dim()}D")

    nope_flat = scores_nope.contiguous().view(B * S, H, T)
    rope_flat = scores_rope.contiguous().view(B * S, H, T)

    if index_mask.dim() == 4:
        mask_3d = index_mask.squeeze(2)
    elif index_mask.dim() == 3:
        mask_3d = index_mask
    elif index_mask.dim() == 2:
        mask_3d = index_mask.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected mask dim {index_mask.dim()}")

    if mask_3d.shape[1] == 1 and S > 1:
        mask_3d = mask_3d.expand(B, S, T)
    mask_flat = mask_3d.contiguous().view(B * S, T)

    BS_total = B * S
    mod = _get_cuda_module()
    out = mod.score_softmax_cuda(
        nope_flat, rope_flat, mask_flat,
        softmax_scale, BS_total, H, T,
    )
    return out.view(orig_shape)
