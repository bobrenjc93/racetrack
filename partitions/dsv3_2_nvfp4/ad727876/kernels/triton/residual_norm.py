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
        residual, update, weight, out_hidden, out_normed,
        eps: tl.constexpr, cols: tl.constexpr,
        stride_r: tl.constexpr, stride_u: tl.constexpr,
        block_size: tl.constexpr,
    ):
        token = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < cols
        r = tl.load(residual + token * stride_r + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(update + token * stride_u + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        hidden = r + u
        tl.store(out_hidden + token * cols + offsets, hidden, mask=mask)
        variance = tl.sum(hidden * hidden, axis=0) / cols
        scale = tl.rsqrt(variance + eps)
        tl.store(out_normed + token * cols + offsets, hidden * scale * w, mask=mask)


def fused_residual_norm(
    update: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float,
    fallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Matches inference model convention: (update, residual) → (normed, hidden)."""
    del fallback
    if update.device.type != "cuda":
        raise RuntimeError("Triton fused_residual_norm requires CUDA tensors")
    shape = update.shape
    cols = shape[-1]
    n_rows = update.numel() // cols
    block_size = triton.next_power_of_2(cols)
    u_flat = update.contiguous().view(n_rows, cols)
    r_flat = residual.contiguous().view(n_rows, cols)
    out_hidden = torch.empty_like(u_flat)
    out_normed = torch.empty_like(u_flat)
    _residual_norm_kernel[(n_rows,)](
        r_flat, u_flat, norm_weight.contiguous(),
        out_hidden, out_normed,
        eps, cols, r_flat.stride(0), u_flat.stride(0), block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return out_normed.view(shape), out_hidden.view(shape)
