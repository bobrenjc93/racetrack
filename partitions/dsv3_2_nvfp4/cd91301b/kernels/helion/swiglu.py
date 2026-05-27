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
    def _swiglu_kernel(
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(gate)
        rows, cols = gate.size()
        for tile_r, tile_c in hl.tile([rows, cols]):
            g = gate[tile_r, tile_c].to(torch.float32)
            u = up[tile_r, tile_c].to(torch.float32)
            silu_g = g * torch.sigmoid(g)
            out[tile_r, tile_c] = (silu_g * u).to(gate.dtype)
        return out


def fused_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    fallback,
) -> torch.Tensor:
    del fallback
    return _swiglu_kernel(gate.contiguous(), up.contiguous())
