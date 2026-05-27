"""FP8 GEMM with tensor cores + fused quant prologue.

Uses tl.dot for H100 FP8 tensor core acceleration. Pads M=1 to BLOCK_M
for tensor core alignment. Fuses input quantization to avoid intermediate.

Based on the original _fp8_gemm_kernel but with inline quantization.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    BACKEND_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    BACKEND_AVAILABLE = False


if BACKEND_AVAILABLE:

    @triton.jit
    def _fp8_gemm_tc_kernel(
        A_bf16_ptr, B_ptr, C_ptr, B_s_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        stride_bs_n, stride_bs_k,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_idx in range(tl.cdiv(K, BLOCK_K)):
            k_start = k_idx * BLOCK_K
            offs_k = k_start + tl.arange(0, BLOCK_K)

            # Load A as bf16 and quantize on-the-fly to fp8
            a_bf16 = tl.load(
                A_bf16_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
            ).to(tl.float32)

            # Per-row amax for this K-block
            a_abs = tl.abs(a_bf16)
            a_amax = tl.max(a_abs, axis=1)  # [BLOCK_M]
            a_amax = tl.where(a_amax > 1e-4, a_amax, 1e-4)
            a_scale = a_amax / 448.0
            a_q = a_bf16 / a_scale[:, None]
            a_fp8 = tl.minimum(tl.maximum(a_q, -448.0), 448.0).to(tl.float8e4nv)

            # Load B as fp8
            b = tl.load(
                B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0,
            )

            # B scale
            b_scale = tl.load(
                B_s_ptr + offs_n * stride_bs_n + k_idx * stride_bs_k,
                mask=offs_n < N, other=1.0,
            )

            # Tensor core FP8 dot product + scale correction
            acc += tl.dot(a_fp8, tl.trans(b)).to(tl.float32) * a_scale[:, None] * b_scale[None, :]

        c = acc.to(tl.bfloat16)
        tl.store(
            C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def fp8_gemm_tc(
    a_bf16: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """FP8 GEMM with tensor cores + fused input quantization."""
    orig_shape = a_bf16.shape
    if a_bf16.dim() > 2:
        a_bf16 = a_bf16.reshape(-1, a_bf16.shape[-1])

    M, K = a_bf16.shape
    N = b_fp8.shape[0]

    if M > 16:
        # Large M: fallback to original (well-tuned for larger M)
        from racetrack.models.deepseek import act_quant, fp8_gemm
        from racetrack.models.deepseek import block_size
        xq, xs = act_quant(a_bf16, block_size)
        return fp8_gemm(xq, xs, b_fp8, b_scale).view(*orig_shape[:-1], N)

    c = torch.empty((M, N), device=a_bf16.device, dtype=torch.bfloat16)
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 128, 128
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _fp8_gemm_tc_kernel[grid](
        a_bf16.contiguous(), b_fp8, c, b_scale,
        M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        b_fp8.stride(0), b_fp8.stride(1),
        c.stride(0), c.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_SIZE_M=8,
        num_warps=4, num_stages=3,
    )
    return c.view(*orig_shape[:-1], N)
