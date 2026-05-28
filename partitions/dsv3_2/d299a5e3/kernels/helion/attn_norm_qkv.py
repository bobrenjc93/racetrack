from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    import helion
    import helion.language as hl

    BACKEND_AVAILABLE = True
except Exception:
    helion = None
    hl = None
    BACKEND_AVAILABLE = False


def _autotune_effort() -> str:
    return os.getenv("RACETRACK_HELION_AUTOTUNE_EFFORT", "none")


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


def fused_attn_norm_qkv(
    hidden_states, norm_weight, qkv_weight,
    *, eps, q_lora_rank, kv_lora_rank, qk_rope_head_dim, fallback,
):
    del fallback
    shape = hidden_states.shape
    h_2d = hidden_states.contiguous().view(-1, shape[-1])
    x = _rms_norm_kernel(h_2d, norm_weight.contiguous(), eps).view(shape)
    qkv = F.linear(x, qkv_weight)
    q_c, kv_c, k_pe = qkv.split(
        [q_lora_rank, kv_lora_rank, qk_rope_head_dim], dim=-1,
    )
    return q_c, kv_c, k_pe
