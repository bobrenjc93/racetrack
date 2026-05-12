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
    def _rms_norm_kernel(
        x,
        weight,
        out,
        eps: tl.constexpr,
        cols: tl.constexpr,
        stride_t: tl.constexpr,
        block_size: tl.constexpr,
    ):
        token = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < cols
        values = tl.load(x + token * stride_t + offsets, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / cols
        scale = tl.rsqrt(variance + eps)
        tl.store(out + token * cols + offsets, values * scale * weights, mask=mask)

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


def _rms_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if not BACKEND_AVAILABLE:
        raise RuntimeError("Triton backend requested, but triton is not installed")
    if x.device.type != "cuda":
        raise RuntimeError("Triton RMSNorm kernel requires CUDA tensors")
    if x.dim() != 2:
        raise RuntimeError("Triton RMSNorm kernel expects a 2D tensor")
    if weight.dim() != 1 or weight.shape[0] != x.shape[-1]:
        raise RuntimeError("Triton RMSNorm weight shape must match the hidden dimension")
    cols = x.shape[-1]
    block_size = triton.next_power_of_2(cols)
    out = torch.empty((x.shape[0], cols), device=x.device, dtype=x.dtype)
    _rms_norm_kernel[(x.shape[0],)](
        x,
        weight.contiguous(),
        out,
        eps,
        cols,
        x.stride(0),
        block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return out


def _apply_rope_triton(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_base: float,
) -> torch.Tensor:
    if not BACKEND_AVAILABLE:
        raise RuntimeError("Triton backend requested, but triton is not installed")
    if x.device.type != "cuda":
        raise RuntimeError("Triton kernels require CUDA tensors")
    if not x.is_contiguous():
        raise RuntimeError("Triton RoPE kernel requires contiguous input")
    if x.dim() != 2:
        raise RuntimeError("Triton RoPE kernel expects a 2D tensor")
    rotary_dim = x.shape[-1]
    if rotary_dim % 2 != 0:
        raise RuntimeError("Triton RoPE kernel requires an even RoPE dimension")
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
    del fallback
    if q_c.device.type != "cuda":
        raise RuntimeError("Triton fused_norm_rope requires CUDA tensors")
    return (
        _rms_norm_triton(q_c, q_weight, eps),
        _rms_norm_triton(kv_c, kv_weight, eps),
        _apply_rope_triton(k_pe.contiguous(), positions, rope_base=rope_base),
    )
