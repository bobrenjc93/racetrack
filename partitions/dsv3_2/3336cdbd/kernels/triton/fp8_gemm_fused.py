"""FP8 GEMM with fused quantization prologue.

Fuses act_quant INTO the GEMM: reads bf16 input, quantizes to fp8
on-the-fly in registers, then does the FP8 matmul. Eliminates the
intermediate fp8 tensor write from act_quant and the subsequent read.

For M=1 decode: saves 2 memory passes over a [1, K] tensor per GEMM call.
With ~10 GEMMs per layer × 27 layers = 270 saved tensor round-trips.
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
    def _fp8_gemm_quant_prologue_kernel(
        A_bf16_ptr, B_ptr, C_ptr, B_s_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bn, stride_bk,
        stride_cm, stride_cn,
        stride_bs_n, stride_bs_k,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    ):
        """
        GEMM with fused input quantization.
        A is bf16 (not fp8) — quantized on-the-fly per BLOCK_K chunk.
        B is fp8 with block scales.
        Output C is bf16.
        """
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

            # Load A as bf16 and quantize to fp8 on-the-fly
            a_bf16 = tl.load(
                A_bf16_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
            ).to(tl.float32)

            # Per-row, per-block quantization
            a_abs = tl.abs(a_bf16)
            a_amax = tl.max(a_abs, axis=1)  # [BLOCK_M]
            a_amax = tl.where(a_amax > 1e-4, a_amax, 1e-4)
            a_scale = a_amax / 448.0  # [BLOCK_M]
            a_scaled = a_bf16 / a_scale[:, None]
            a = tl.minimum(tl.maximum(a_scaled, -448.0), 448.0).to(tl.float8e4nv)

            # Load B (already fp8)
            b = tl.load(
                B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0,
            )

            # Load B scale
            b_scale = tl.load(
                B_s_ptr + offs_n * stride_bs_n + k_idx * stride_bs_k,
                mask=offs_n < N, other=1.0,
            )

            # FP8 dot product with post-multiply scales.
            # BLOCK_K=128 matches quant block_size, so each k_idx tile
            # has uniform a_scale per row and b_scale per column.
            acc += tl.dot(a, tl.trans(b)).to(tl.float32) * a_scale[:, None] * b_scale[None, :]

        # Store result as bf16
        c = acc.to(tl.bfloat16)
        tl.store(
            C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def fp8_gemm_fused(
    a_bf16: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 GEMM with fused input quantization.
    a_bf16: [M, K] bf16 input (NOT pre-quantized)
    b_fp8: [N, K] fp8 weight
    b_scale: [N, K//128] float32 weight scale
    Returns: [M, N] bf16
    """
    orig_shape = a_bf16.shape
    if a_bf16.dim() > 2:
        a_bf16 = a_bf16.reshape(-1, a_bf16.shape[-1])

    M, K = a_bf16.shape
    N = b_fp8.shape[0]
    c = torch.empty((M, N), device=a_bf16.device, dtype=torch.bfloat16)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 128
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _fp8_gemm_quant_prologue_kernel[grid](
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
