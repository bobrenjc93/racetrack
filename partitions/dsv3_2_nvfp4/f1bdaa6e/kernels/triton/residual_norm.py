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
    residual: torch.Tensor,
    update: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float,
    fallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    del fallback
    if residual.device.type != "cuda":
        raise RuntimeError("Triton fused_residual_norm requires CUDA tensors")
    tokens = residual.shape[0]
    cols = residual.shape[-1]
    block_size = triton.next_power_of_2(cols)
    out_hidden = torch.empty((tokens, cols), device=residual.device, dtype=residual.dtype)
    out_normed = torch.empty((tokens, cols), device=residual.device, dtype=residual.dtype)
    _residual_norm_kernel[(tokens,)](
        residual.contiguous(), update.contiguous(), norm_weight.contiguous(),
        out_hidden, out_normed,
        eps, cols, residual.stride(0), update.stride(0), block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return out_hidden, out_normed
