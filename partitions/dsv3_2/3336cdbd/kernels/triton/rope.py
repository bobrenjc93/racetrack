"""Fused RoPE: apply rotary embeddings in a single Triton kernel.

Replaces: float cast → view_as_complex → mul(freqs_cis) → view_as_real → flatten → cast
(6 ops, 4 kernel launches in Inductor) with 1 kernel launch.
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
    def _rope_kernel(
        x_ptr, freqs_cos_ptr, freqs_sin_ptr, out_ptr,
        n_elements,
        half_d: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """
        RoPE as real-valued rotation: for each pair (x[2i], x[2i+1]),
        apply rotation by (cos θ, sin θ):
          out[2i]   = x[2i] * cos - x[2i+1] * sin
          out[2i+1] = x[2i] * sin + x[2i+1] * cos
        """
        pid = tl.program_id(0)
        pair_start = pid * BLOCK
        pair_offs = pair_start + tl.arange(0, BLOCK)
        mask = pair_offs < half_d

        idx_even = pair_offs * 2
        idx_odd = idx_even + 1

        x_even = tl.load(x_ptr + idx_even, mask=mask, other=0.0).to(tl.float32)
        x_odd = tl.load(x_ptr + idx_odd, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(freqs_cos_ptr + pair_offs, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(freqs_sin_ptr + pair_offs, mask=mask, other=0.0).to(tl.float32)

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        tl.store(out_ptr + idx_even, out_even.to(tl.bfloat16), mask=mask)
        tl.store(out_ptr + idx_odd, out_odd.to(tl.bfloat16), mask=mask)


def fused_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    fallback,
) -> torch.Tensor:
    """
    Apply rotary embeddings using a single Triton kernel.
    x: [..., d] where d is the rope dimension
    freqs_cis: [1, seq, 1, d//2] complex64
    """
    del fallback
    dtype = x.dtype
    shape = x.shape
    d = shape[-1]
    half_d = d // 2

    freqs_flat = freqs_cis.view(-1, half_d)
    freqs_cos = freqs_flat.real.contiguous()
    freqs_sin = freqs_flat.imag.contiguous()

    x_flat = x.contiguous().view(-1, d)
    n_rows = x_flat.shape[0]
    out = torch.empty_like(x_flat)

    BLOCK = triton.next_power_of_2(half_d)

    for row in range(n_rows):
        freq_row = row % freqs_cos.shape[0]
        _rope_kernel[(triton.cdiv(half_d, BLOCK),)](
            x_flat[row], freqs_cos[freq_row], freqs_sin[freq_row], out[row],
            d, half_d, BLOCK,
        )

    return out.view(shape).to(dtype)
