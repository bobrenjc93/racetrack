"""Inline RoPE: bf16 input → rotary embedding → bf16 output in one kernel.

Replaces: .float() → view_as_complex → mul(freqs) → view_as_real → flatten → .to(bf16)
(6 ops, each a separate kernel) with 1 Triton kernel.

Uses real-valued rotation (no complex math):
  out[2i]   = x[2i]*cos - x[2i+1]*sin
  out[2i+1] = x[2i]*sin + x[2i+1]*cos
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
    def _rope_inline_kernel(
        x_ptr, cos_ptr, sin_ptr, out_ptr,
        n_pairs: tl.constexpr,
        stride_row,
        BLOCK: tl.constexpr,
    ):
        """One program per row. Applies RoPE to n_pairs (x[2i], x[2i+1]) pairs."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_pairs

        idx_even = offs * 2
        idx_odd = idx_even + 1

        x_e = tl.load(x_ptr + row * stride_row + idx_even, mask=mask, other=0.0).to(tl.float32)
        x_o = tl.load(x_ptr + row * stride_row + idx_odd, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        s = tl.load(sin_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        out_e = x_e * c - x_o * s
        out_o = x_e * s + x_o * c

        tl.store(out_ptr + row * stride_row + idx_even, out_e.to(tl.bfloat16), mask=mask)
        tl.store(out_ptr + row * stride_row + idx_odd, out_o.to(tl.bfloat16), mask=mask)


def rope_inline(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings: bf16 in → bf16 out, 1 kernel.
    x: [..., d] where d is the rope dim
    freqs_cis: [1, seq, 1, d//2] complex64
    """
    shape = x.shape
    d = shape[-1]
    n_pairs = d // 2

    # Extract cos/sin from complex freqs
    freqs_flat = freqs_cis.view(-1, n_pairs)
    cos = freqs_flat.real.contiguous()
    sin = freqs_flat.imag.contiguous()

    x_flat = x.contiguous().view(-1, d)
    n_rows = x_flat.shape[0]
    out = torch.empty_like(x_flat)

    BLOCK = triton.next_power_of_2(n_pairs)

    for r in range(n_rows):
        freq_r = r % cos.shape[0]
        _rope_inline_kernel[(1,)](
            x_flat[r:r+1].view(-1), cos[freq_r], sin[freq_r], out[r:r+1].view(-1),
            n_pairs, d, BLOCK,
        )

    return out.view(shape)
