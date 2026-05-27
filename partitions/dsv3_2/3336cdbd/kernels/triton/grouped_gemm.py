"""Grouped FP8 GEMV: all topk experts in one kernel launch.

For single-token decode: x [1, K] @ W[topk, N, K].T -> out [topk, N]
Each program handles one expert's one output row.
Grid: (topk * N,) — one program per (expert, output_row).
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
    def _grouped_fp8_gemv_kernel(
        x_ptr, x_scale_ptr,
        w_ptr, w_scale_ptr,
        out_ptr,
        K: tl.constexpr,
        N,
        w_stride_expert,
        ws_stride_expert,
        n_k_blocks: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        One program = one (expert, output_row) pair.
        Computes: out[expert, row] = sum_k x[k] * W[expert, row, k] * x_scale * w_scale
        """
        pid = tl.program_id(0)
        expert_id = pid // N
        row = pid % N

        acc = tl.zeros([], dtype=tl.float32)

        for kb in range(n_k_blocks):
            k_start = kb * BLOCK_K
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K

            x_vals = tl.load(x_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            x_s = tl.load(x_scale_ptr + kb).to(tl.float32)

            w_vals = tl.load(
                w_ptr + expert_id * w_stride_expert + row * K + k_offs,
                mask=k_mask, other=0.0,
            ).to(tl.float32)
            w_s = tl.load(
                w_scale_ptr + expert_id * ws_stride_expert + row * n_k_blocks + kb,
            ).to(tl.float32)

            acc += tl.sum(x_vals * w_vals, axis=0) * x_s * w_s

        tl.store(out_ptr + expert_id * N + row, acc.to(tl.bfloat16))


def grouped_fp8_gemv(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_stacked: torch.Tensor,
    w_scale_stacked: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """
    x [1, K] @ W[n_experts, N, K].T -> [n_experts, N]
    Single kernel launch for all experts.
    """
    K = x_fp8.shape[-1]
    N = w_stacked.shape[1]
    BLOCK_K = 128
    n_k_blocks = (K + BLOCK_K - 1) // BLOCK_K

    out = torch.empty(n_experts, N, device=x_fp8.device, dtype=torch.bfloat16)

    grid = (n_experts * N,)
    _grouped_fp8_gemv_kernel[grid](
        x_fp8.view(-1),
        x_scale.view(-1),
        w_stacked,
        w_scale_stacked,
        out,
        K, N,
        w_stacked.stride(0),
        w_scale_stacked.stride(0),
        n_k_blocks,
        BLOCK_K,
    )
    return out
