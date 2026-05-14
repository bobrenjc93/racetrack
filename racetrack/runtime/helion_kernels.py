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


def _require_helion_cuda(*tensors: torch.Tensor) -> None:
    if not BACKEND_AVAILABLE:
        raise RuntimeError("Helion backend requested, but helion is not installed")
    if not tensors or any(tensor.device.type != "cuda" for tensor in tensors):
        raise RuntimeError("Helion kernels require CUDA tensors")


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
            hidden = residual[tile_t, :].to(torch.float32) + update[tile_t, :].to(torch.float32)
            out_hidden[tile_t, :] = hidden.to(residual.dtype)
            variance = torch.mean(hidden * hidden, dim=1)
            scale = torch.rsqrt(variance + eps).view(tile_t, 1)
            out_normed[tile_t, :] = (
                hidden * scale * weight[:].to(torch.float32)
            ).to(residual.dtype)
        return out_hidden, out_normed

    @helion.kernel(autotune_effort=_autotune_effort())
    def _swiglu_kernel(
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(gate)
        rows, cols = gate.size()
        for tile_r, tile_c in hl.tile([rows, cols]):
            gate_values = gate[tile_r, tile_c].to(torch.float32)
            up_values = up[tile_r, tile_c].to(torch.float32)
            silu_gate = gate_values * torch.sigmoid(gate_values)
            out[tile_r, tile_c] = (silu_gate * up_values).to(gate.dtype)
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
    _require_helion_cuda(q_c, q_weight, kv_c, kv_weight, k_pe, positions)
    if q_c.dim() != 2 or kv_c.dim() != 2 or k_pe.dim() != 2:
        raise RuntimeError("Helion fused_norm_rope expects 2D tensors")
    if k_pe.shape[-1] % 2 != 0:
        raise RuntimeError("Helion fused_norm_rope requires an even RoPE dimension")
    return (
        _rms_norm_kernel(q_c.contiguous(), q_weight.contiguous(), eps),
        _rms_norm_kernel(kv_c.contiguous(), kv_weight.contiguous(), eps),
        _rope_kernel(k_pe.contiguous(), positions.contiguous(), math.log(rope_base)),
    )


def fused_residual_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float,
    fallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    del fallback
    _require_helion_cuda(residual, update, norm_weight)
    if residual.dim() != 2 or update.dim() != 2:
        raise RuntimeError("Helion fused_residual_norm expects 2D tensors")
    if residual.shape != update.shape:
        raise RuntimeError("Helion residual and update shapes must match")
    if norm_weight.dim() != 1 or norm_weight.shape[0] != residual.shape[-1]:
        raise RuntimeError("Helion norm weight shape must match hidden dimension")
    return _residual_norm_kernel(
        residual.contiguous(),
        update.contiguous(),
        norm_weight.contiguous(),
        eps,
    )


def fused_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    fallback,
) -> torch.Tensor:
    del fallback
    _require_helion_cuda(gate, up)
    if gate.dim() != 2:
        raise RuntimeError("Helion fused_swiglu expects 2D tensors")
    if gate.shape != up.shape:
        raise RuntimeError("Helion fused_swiglu inputs must have matching shapes")
    return _swiglu_kernel(gate.contiguous(), up.contiguous())
