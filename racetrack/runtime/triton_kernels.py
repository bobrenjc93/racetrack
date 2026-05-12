from __future__ import annotations

import torch

from . import torch_ops

try:
    import triton
    import triton.language as tl

    BACKEND_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    BACKEND_AVAILABLE = False


if BACKEND_AVAILABLE:

    @triton.jit
    def _rope_kernel(x, cos, sin, out, rotary_dim: tl.constexpr, stride_t: tl.constexpr):
        token = tl.program_id(0)
        half: tl.constexpr = rotary_dim // 2
        offsets = tl.arange(0, half)
        row = token * stride_t
        x1 = tl.load(x + row + offsets).to(tl.float32)
        x2 = tl.load(x + row + offsets + half).to(tl.float32)
        c = tl.load(cos + token * half + offsets).to(tl.float32)
        s = tl.load(sin + token * half + offsets).to(tl.float32)
        tl.store(out + row + offsets, x1 * c - x2 * s)
        tl.store(out + row + offsets + half, x2 * c + x1 * s)


def _apply_rope_triton(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_base: float,
) -> torch.Tensor:
    if not BACKEND_AVAILABLE or x.device.type != "cuda" or not x.is_contiguous():
        return torch_ops.apply_rope(x, positions, base=rope_base)
    if x.dim() != 2:
        return torch_ops.apply_rope(x, positions, base=rope_base)
    rotary_dim = x.shape[-1]
    if rotary_dim % 2 != 0:
        return torch_ops.apply_rope(x, positions, base=rope_base)
    cos, sin = torch_ops.rope_cache(positions, rotary_dim, base=rope_base, dtype=x.dtype)
    out = torch.empty_like(x)
    _rope_kernel[(x.shape[0],)](
        x,
        cos.contiguous(),
        sin.contiguous(),
        out,
        rotary_dim,
        x.stride(0),
        num_warps=1,
    )
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
    if not BACKEND_AVAILABLE or q_c.device.type != "cuda":
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
    return (
        torch_ops.rms_norm(q_c, q_weight, eps),
        torch_ops.rms_norm(kv_c, kv_weight, eps),
        _apply_rope_triton(k_pe.contiguous(), positions, rope_base=rope_base),
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
