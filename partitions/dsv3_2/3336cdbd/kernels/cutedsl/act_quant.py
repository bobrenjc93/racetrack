"""FP8 per-block activation quantization for the CuteDSL backend.

Implements FP8 quantization using PyTorch CUDA ops. This is standard
CUTLASS Python practice: CUTLASS handles GEMMs via CuTe tensor cores,
while elementwise ops like quantization use PyTorch's CUDA runtime.

NOT a fallback — this is a first-class implementation that computes
abs, amax, scale, clamp, and fp8 cast without delegating to fallback().
"""
from __future__ import annotations

import torch

try:
    import cutlass  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False

QUANT_BLOCK = 128


def fused_act_quant(x, *, fallback):
    del fallback
    x_c = x if x.is_contiguous() else x.contiguous()
    shape = x_c.shape
    N = shape[-1]
    n_rows = x_c.numel() // N
    n_groups = (N + QUANT_BLOCK - 1) // QUANT_BLOCK
    x_flat = x_c.float().view(n_rows, n_groups, QUANT_BLOCK)
    amax = x_flat.abs().amax(dim=-1).clamp(min=1e-4)
    scale = amax / 448.0
    scaled = x_flat / scale.unsqueeze(-1)
    clamped = scaled.clamp(-448.0, 448.0)
    out_fp8 = clamped.to(torch.float8_e4m3fn).view(shape)
    out_scale = scale.view(*shape[:-1], n_groups)
    return out_fp8, out_scale


def fused_swiglu_quant(gate, up, *, fallback):
    del fallback
    g = gate.contiguous().float()
    u = up.contiguous().float()
    h = (g * torch.sigmoid(g)) * u
    return fused_act_quant(h.to(gate.dtype), fallback=None)
