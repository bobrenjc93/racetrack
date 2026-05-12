from __future__ import annotations

import importlib.util

import torch


def package_available(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


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
    return fallback(
        q_c,
        q_weight,
        kv_c,
        kv_weight,
        k_pe,
        positions,
        eps=eps,
        rope_base=rope_base,
    )


def hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_norm_eps: float,
    hc_eps: float,
    fallback,
) -> torch.Tensor:
    return fallback(
        hidden_states,
        hc_fn,
        hc_scale,
        hc_base,
        rms_norm_eps=rms_norm_eps,
        hc_eps=hc_eps,
    )
