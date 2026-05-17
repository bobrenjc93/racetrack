"""FP8 GEMV with fused quant prologue, optimized for M=1 decode.

One program per output element. Each program:
1. Loads x (bf16) in BLOCK_K chunks
2. Quantizes to fp8 on-the-fly (per-block scale)
3. Dots with one row of W (fp8)
4. Accumulates with scale correction
5. Writes one bf16 output

The fp8 intermediate never hits HBM.
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
    def _fp8_gemv_fused_kernel(
        x_ptr, w_ptr, w_scale_ptr, out_ptr,
        K: tl.constexpr,
        stride_wn,
        stride_wsn,
        n_k_blocks: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One program = one output element."""
        n_idx = tl.program_id(0)
        acc = tl.zeros([], dtype=tl.float32)

        for kb in range(n_k_blocks):
            k_start = kb * BLOCK_K
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K

            # Load x block, quantize on-the-fly
            x_vals = tl.load(x_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            x_amax = tl.max(tl.abs(x_vals), axis=0)
            x_amax = tl.where(x_amax > 1e-4, x_amax, 1e-4)
            x_scale = x_amax / 448.0
            x_q = tl.minimum(tl.maximum(x_vals / x_scale, -448.0), 448.0)

            # Load W row and scale
            w_vals = tl.load(
                w_ptr + n_idx * stride_wn + k_offs,
                mask=k_mask, other=0.0,
            ).to(tl.float32)
            w_s = tl.load(w_scale_ptr + n_idx * stride_wsn + kb)

            # Dot product with scale correction
            acc += tl.sum(x_q * w_vals, axis=0) * x_scale * w_s

        tl.store(out_ptr + n_idx, acc.to(tl.bfloat16))


def fp8_gemv_fused(
    x_bf16: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """
    M=1 FP8 GEMV with fused input quantization.
    x_bf16: [*, K] bf16
    w_fp8: [N, K] fp8
    w_scale: [N, K//128] float32
    Returns: [*, N] bf16
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

    _fp8_gemv_fused_kernel[(N,)](
        x_flat.view(-1), w_fp8, w_scale, out,
        K,
        w_fp8.stride(0),
        w_scale.stride(0),
        n_k_blocks,
        128,
    )
    return out.view(*orig_shape[:-1], N)
