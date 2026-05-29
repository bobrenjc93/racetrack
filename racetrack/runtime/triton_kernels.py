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
    def _rope_kernel(
        x,
        cos,
        sin,
        out,
        rotary_dim: tl.constexpr,
        stride_t: tl.constexpr,
        block_size: tl.constexpr,
    ):
        token = tl.program_id(0)
        half: tl.constexpr = rotary_dim // 2
        offsets = tl.arange(0, block_size)
        mask = offsets < half
        row = token * stride_t
        x1 = tl.load(x + row + offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x + row + offsets + half, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos + token * half + offsets, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(sin + token * half + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out + row + offsets, x1 * c - x2 * s, mask=mask)
        tl.store(out + row + offsets + half, x2 * c + x1 * s, mask=mask)

    @triton.jit
    def _residual_norm_kernel(
        residual,
        update,
        weight,
        out_hidden,
        out_normed,
        eps: tl.constexpr,
        cols: tl.constexpr,
        residual_stride_t: tl.constexpr,
        update_stride_t: tl.constexpr,
        block_size: tl.constexpr,
    ):
        token = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < cols
        r = tl.load(
            residual + token * residual_stride_t + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        u = tl.load(
            update + token * update_stride_t + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        hidden = r + u
        tl.store(out_hidden + token * cols + offsets, hidden, mask=mask)
        variance = tl.sum(hidden * hidden, axis=0) / cols
        scale = tl.rsqrt(variance + eps)
        tl.store(out_normed + token * cols + offsets, hidden * scale * w, mask=mask)

    @triton.jit
    def _swiglu_kernel(
        gate,
        up,
        out,
        n_elements: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        gate_values = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
        up_values = tl.load(up + offsets, mask=mask, other=0.0).to(tl.float32)
        silu_gate = gate_values * tl.sigmoid(gate_values)
        tl.store(out + offsets, silu_gate * up_values, mask=mask)


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
    block_size = triton.next_power_of_2(rotary_dim // 2)
    _rope_kernel[(x.shape[0],)](
        x,
        cos.contiguous(),
        sin.contiguous(),
        out,
        rotary_dim,
        x.stride(0),
        block_size,
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


def fused_residual_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float,
    fallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    del fallback
    if not BACKEND_AVAILABLE:
        raise RuntimeError("Triton backend requested, but triton is not installed")
    if residual.device.type != "cuda":
        raise RuntimeError("Triton fused_residual_norm requires CUDA tensors")
    if residual.dim() != 2 or update.dim() != 2:
        raise RuntimeError("Triton fused_residual_norm expects 2D tensors")
    if residual.shape != update.shape:
        raise RuntimeError("Triton residual and update shapes must match")
    if norm_weight.dim() != 1 or norm_weight.shape[0] != residual.shape[-1]:
        raise RuntimeError("Triton norm weight shape must match hidden dimension")
    tokens, cols = residual.shape
    block_size = triton.next_power_of_2(cols)
    residual_c = residual.contiguous()
    update_c = update.contiguous()
    out_hidden = torch.empty((tokens, cols), device=residual.device, dtype=residual.dtype)
    out_normed = torch.empty((tokens, cols), device=residual.device, dtype=residual.dtype)
    _residual_norm_kernel[(tokens,)](
        residual_c,
        update_c,
        norm_weight.contiguous(),
        out_hidden,
        out_normed,
        float(eps),
        cols,
        residual_c.stride(0),
        update_c.stride(0),
        block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return out_hidden, out_normed


def fused_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    fallback,
) -> torch.Tensor:
    del fallback
    if not BACKEND_AVAILABLE:
        raise RuntimeError("Triton backend requested, but triton is not installed")
    if gate.device.type != "cuda":
        raise RuntimeError("Triton fused_swiglu requires CUDA tensors")
    if gate.shape != up.shape:
        raise RuntimeError("Triton fused_swiglu inputs must have matching shapes")
    gate_c = gate.contiguous()
    up_c = up.contiguous()
    out = torch.empty_like(gate_c)
    block_size = 1024
    grid = (triton.cdiv(gate_c.numel(), block_size),)
    _swiglu_kernel[grid](
        gate_c,
        up_c,
        out,
        gate_c.numel(),
        block_size,
        num_warps=4,
    )
    return out.view_as(gate)
