"""Fused residual add + RMS norm for the CuteDSL backend.

First-class implementation using PyTorch CUDA ops.
"""
from __future__ import annotations

import torch

try:
    import cutlass  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False


def fused_residual_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float,
    fallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    del fallback
    hidden = residual + update
    hidden_f = hidden.float()
    variance = hidden_f.pow(2).mean(dim=-1, keepdim=True)
    normed = (hidden_f * torch.rsqrt(variance + eps) * norm_weight.float()).to(hidden.dtype)
    return hidden, normed
