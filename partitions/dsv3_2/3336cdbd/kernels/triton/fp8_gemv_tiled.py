"""Tiled FP8 GEMV with fused quant prologue for M=1 decode.

Grid: (cdiv(N, BLOCK_N),)
Each program handles BLOCK_N output elements, sharing the input x across them.
Input x is quantized on-the-fly per K-block in registers.

This is the correct architecture: tiles over N (sharing x in L2),
fuses quant (no fp8 intermediate in HBM), optimized for M=1.
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
    def _fp8_gemv_tiled_kernel(
        x_ptr, w_ptr, w_scale_ptr, out_ptr,
        N, K: tl.constexpr,
        stride_wn, stride_wk: tl.constexpr,
        stride_wsn, stride_wsk: tl.constexpr,
        n_k_blocks: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Each program: BLOCK_N outputs, iterating over K in BLOCK_K chunks.
        Input x is loaded ONCE per K-block and reused for all BLOCK_N outputs.
        """
        pid = tl.program_id(0)
        n_start = pid * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        for kb in range(n_k_blocks):
            k_start = kb * BLOCK_K
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K

            # Load x once for this K-block, quantize on-the-fly
            x_vals = tl.load(x_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            x_amax = tl.max(tl.abs(x_vals), axis=0)
            x_amax = tl.where(x_amax > 1e-4, x_amax, 1e-4)
            x_scale = x_amax / 448.0
            x_q = tl.minimum(tl.maximum(x_vals / x_scale, -448.0), 448.0)

            # Load W block: [BLOCK_N, BLOCK_K]
            w_block = tl.load(
                w_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # Load W scales: [BLOCK_N]
            w_s = tl.load(
                w_scale_ptr + n_offs * stride_wsn + kb * stride_wsk,
                mask=n_mask, other=1.0,
            )

            # Dot: each output = sum(x_q * w_row) * x_scale * w_scale
            dot = tl.sum(x_q[None, :] * w_block, axis=1)  # [BLOCK_N]
            acc += dot * x_scale * w_s

        tl.store(out_ptr + n_offs, acc.to(tl.bfloat16), mask=n_mask)


def fp8_gemv_tiled(
    x_bf16: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Tiled M=1 FP8 GEMV with fused input quantization.
    Falls back to regular path for M>1.
    """
    orig_shape = x_bf16.shape
    K = orig_shape[-1]
    x_flat = x_bf16.contiguous().view(-1, K)

    if x_flat.shape[0] != 1:
        from inference.kernel import act_quant, fp8_gemm
        from inference.model import block_size
        xq, xs = act_quant(x_flat, block_size)
        return fp8_gemm(xq, xs, w_fp8, w_scale).view(*orig_shape[:-1], w_fp8.shape[0])

    N = w_fp8.shape[0]
    n_k_blocks = (K + 127) // 128
    out = torch.empty(N, device=x_bf16.device, dtype=torch.bfloat16)

    BLOCK_N = 64
    BLOCK_K = 128
    grid = (triton.cdiv(N, BLOCK_N),)

    _fp8_gemv_tiled_kernel[grid](
        x_flat.view(-1), w_fp8, w_scale, out,
        N, K,
        w_fp8.stride(0), w_fp8.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        n_k_blocks,
        BLOCK_N, BLOCK_K,
    )
    return out.view(*orig_shape[:-1], N)
