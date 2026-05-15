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
    def _residual_norm_kernel(
        x, residual, weight, out_normed, out_hidden,
        eps: tl.constexpr, cols: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < cols
        x_values = tl.load(x + row * cols + offsets, mask=mask, other=0.0).to(tl.float32)
        r_values = tl.load(residual + row * cols + offsets, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        hidden = x_values + r_values
        tl.store(out_hidden + row * cols + offsets, hidden, mask=mask)
        variance = tl.sum(hidden * hidden, axis=0) / cols
        scale = tl.rsqrt(variance + eps)
        tl.store(out_normed + row * cols + offsets, hidden * scale * weights, mask=mask)


def fused_residual_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    fallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    del fallback
    x_c = x.contiguous()
    residual_c = residual.contiguous()
    shape = x_c.shape
    cols = shape[-1]
    rows = x_c.numel() // cols
    x_flat = x_c.view(rows, cols)
    residual_flat = residual_c.view(rows, cols)
    out_normed = torch.empty_like(x_flat)
    out_hidden = torch.empty_like(x_flat)
    block_size = triton.next_power_of_2(cols)
    _residual_norm_kernel[(rows,)](
        x_flat, residual_flat, weight.contiguous(),
        out_normed, out_hidden,
        eps, cols, block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return out_normed.view(shape), out_hidden.view(shape)
