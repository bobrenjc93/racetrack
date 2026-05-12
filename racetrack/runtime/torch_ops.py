from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x_float = x.float()
    scale = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    return (x_float * scale * weight.float()).to(orig_dtype)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def rope_cache(
    positions: torch.Tensor,
    rotary_dim: int,
    *,
    base: float = 10000.0,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    device = positions.device
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, half, device=device, dtype=torch.float32) / max(half, 1))
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int | None = None,
    base: float = 10000.0,
) -> torch.Tensor:
    rotary_dim = x.shape[-1] if rotary_dim is None else rotary_dim
    if rotary_dim == 0:
        return x
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"rotary_dim {rotary_dim} exceeds tensor dim {x.shape[-1]}")
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")

    cos, sin = rope_cache(positions, rotary_dim, base=base, dtype=x.dtype)
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    while cos.dim() < x1.dim():
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    if x_pass.numel() == 0:
        return rotated
    return torch.cat((rotated, x_pass), dim=-1)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        rms_norm(q_c, q_weight, eps),
        rms_norm(kv_c, kv_weight, eps),
        apply_rope(k_pe, positions, base=rope_base),
    )


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    scores = torch.einsum("thd,shd->hts", q.float(), k.float()) * scale
    tokens = q.shape[0]
    mask = torch.ones(tokens, tokens, device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask.unsqueeze(0), -float("inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.einsum("hts,shd->thd", probs, v)


def swiglu(gate: torch.Tensor, up: torch.Tensor, *, clamp: float | None = None) -> torch.Tensor:
    if clamp is not None:
        gate = gate.clamp(min=-clamp, max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    return F.silu(gate) * up


def hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    shape = hidden_states.shape
    dtype = hidden_states.dtype
    x = hidden_states.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x, hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * hidden_states.float().view(shape), dim=1)
    return y.to(dtype)


def hc_pre(
    hidden_states: torch.Tensor,
    mix_weight: torch.Tensor,
    mix_scale: torch.Tensor,
    mix_base: torch.Tensor,
    *,
    rms_norm_eps: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mixed = hc_head(
        hidden_states,
        mix_weight,
        mix_scale,
        mix_base,
        rms_norm_eps=rms_norm_eps,
        hc_eps=hc_eps,
    )
    return rms_norm(mixed, torch.ones(mixed.shape[-1], device=mixed.device, dtype=mixed.dtype), rms_norm_eps), mixed


def hc_post(
    update: torch.Tensor,
    residual: torch.Tensor,
    residual_mix: torch.Tensor,
    stream_scale: torch.Tensor,
) -> torch.Tensor:
    del residual_mix
    scale = stream_scale.view(1, -1, 1).to(update.dtype)
    return residual + update.unsqueeze(1) * scale


def stable_randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    scale: float = 1.0,
) -> torch.Tensor:
    fan_in = shape[-1] if shape else 1
    std = scale / math.sqrt(max(fan_in, 1))
    return torch.randn(shape, generator=generator) * std
