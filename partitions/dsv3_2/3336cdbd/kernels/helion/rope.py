from __future__ import annotations

import torch

try:
    import helion  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False


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
    cos = freqs_flat.real.contiguous().to(torch.float32)
    sin = freqs_flat.imag.contiguous().to(torch.float32)

    x_flat = x.contiguous().view(-1, d)
    n_rows = x_flat.shape[0]

    x1 = x_flat[:, :half_d].to(torch.float32)
    x2 = x_flat[:, half_d:].to(torch.float32)
    freq_rows = cos.shape[0]
    if n_rows != freq_rows:
        indices = torch.arange(n_rows, device=cos.device) % freq_rows
        cos = cos[indices]
        sin = sin[indices]

    out_even = x1 * cos - x2 * sin
    out_odd = x1 * sin + x2 * cos
    out = torch.cat([out_even, out_odd], dim=-1).to(x.dtype)
    return out.view(shape)
