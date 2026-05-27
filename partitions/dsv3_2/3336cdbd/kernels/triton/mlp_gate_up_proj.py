from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False


_CAT_CACHE = {}


def _cached_cat(*tensors):
    if any(tensor is None for tensor in tensors):
        return None
    if torch.is_grad_enabled():
        return torch.cat(tensors, dim=0)
    key = tuple(
        (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
            getattr(tensor, "_version", 0),
        )
        for tensor in tensors
    )
    cached = _CAT_CACHE.get(key)
    if cached is None:
        cached = torch.cat(tensors, dim=0).contiguous()
        _CAT_CACHE[key] = cached
    return cached


def fused_mlp_gate_up_proj(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w3_weight: torch.Tensor,
    w3_scale: torch.Tensor | None,
    *,
    scale_fmt: str | None,
    fallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    del fallback
    gate_features = w1_weight.shape[0]
    if w1_weight.dtype != torch.float8_e4m3fn:
        gate_up = F.linear(x, _cached_cat(w1_weight, w3_weight))
    else:
        from racetrack.models import deepseek as real_model

        x_fp8, x_scale = real_model.act_quant(x, real_model.block_size, scale_fmt)
        gate_up = real_model.fp8_gemm(
            x_fp8,
            x_scale,
            _cached_cat(w1_weight, w3_weight),
            _cached_cat(w1_scale, w3_scale),
        )
    return torch.split(gate_up, [gate_features, w3_weight.shape[0]], dim=-1)
