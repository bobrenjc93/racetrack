from __future__ import annotations

import math
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
    def _rms_norm_kernel(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        out = torch.empty_like(x)
        tokens, _hidden = x.size()
        for tile_t in hl.tile(tokens):
            values = x[tile_t, :]
            values_f32 = values.to(torch.float32)
            variance = torch.mean(values_f32 * values_f32, dim=1)
            scale = torch.rsqrt(variance + eps).view(tile_t, 1)
            out[tile_t, :] = (
                values_f32 * scale * weight[:].to(torch.float32)
            ).to(x.dtype)
        return out

    @helion.kernel(autotune_effort=_autotune_effort())
    def _rope_kernel(
        x: torch.Tensor,
        positions: torch.Tensor,
        log_rope_base: float,
    ) -> torch.Tensor:
        out = torch.empty_like(x)
        tokens, rotary_dim = x.size()
        half = rotary_dim // 2
        for tile_t, tile_h in hl.tile([tokens, half]):
            rotary_index = tile_h.index.to(torch.float32)
            position_values = positions[tile_t].to(torch.float32).view(tile_t, 1)
            inv_freq = torch.exp(-(rotary_index / half) * log_rope_base)
            freqs = position_values * inv_freq.view(1, tile_h)
            cos = torch.cos(freqs).to(x.dtype)
            sin = torch.sin(freqs).to(x.dtype)
            x1 = x[tile_t, tile_h]
            x2 = x[tile_t, tile_h + half]
            out[tile_t, tile_h] = (x1 * cos - x2 * sin).to(x.dtype)
            out[tile_t, tile_h + half] = (x2 * cos + x1 * sin).to(x.dtype)
        return out


def fused_norm_rope(
    q_c: torch.Tensor,
    q_weight: torch.Tensor,
    kv_c: torch.Tensor,
    kv_weight: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
    *,
    eps: float,
    rope_base: float,
    fallback,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del fallback
    return (
        _rms_norm_kernel(q_c.contiguous(), q_weight.contiguous(), eps),
        _rms_norm_kernel(kv_c.contiguous(), kv_weight.contiguous(), eps),
        _rope_kernel(k_pe.contiguous(), positions.contiguous(), math.log(rope_base)),
    )
