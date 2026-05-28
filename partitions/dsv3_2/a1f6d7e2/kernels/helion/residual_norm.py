from __future__ import annotations

import os

import torch

try:
    import helion
    import helion.language as hl

    BACKEND_AVAILABLE = True
except Exception:
    helion = None
    hl = None
    BACKEND_AVAILABLE = False


def _autotune_effort() -> str:
    return os.getenv("RACETRACK_HELION_AUTOTUNE_EFFORT", "quick")


if BACKEND_AVAILABLE:

    @helion.kernel(autotune_effort=_autotune_effort())
    def _residual_norm_kernel(
        residual: torch.Tensor,
        update: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_hidden = torch.empty_like(residual)
        out_normed = torch.empty_like(residual)
        tokens, _hidden = residual.size()
        for tile_t in hl.tile(tokens):
            r = residual[tile_t, :].to(torch.float32)
            u = update[tile_t, :].to(torch.float32)
            hidden = r + u
            out_hidden[tile_t, :] = hidden.to(residual.dtype)
            variance = torch.mean(hidden * hidden, dim=1)
            scale = torch.rsqrt(variance + eps).view(tile_t, 1)
            normed = hidden * scale * weight[:].to(torch.float32)
            out_normed[tile_t, :] = normed.to(residual.dtype)
        return out_hidden, out_normed


def fused_residual_norm(
    residual, update, norm_weight,
    *, eps, fallback,
):
    del fallback
    shape = residual.shape
    cols = shape[-1]
    n_rows = residual.numel() // cols
    hidden, normed = _residual_norm_kernel(
        residual.contiguous().view(n_rows, cols),
        update.contiguous().view(n_rows, cols),
        norm_weight.contiguous(), eps,
    )
    return normed.view(shape), hidden.view(shape)
