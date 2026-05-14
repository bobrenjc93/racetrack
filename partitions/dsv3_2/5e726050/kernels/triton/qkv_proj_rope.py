from __future__ import annotations

import torch
import torch.nn.functional as F

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
        x, weight, out,
        eps: tl.constexpr, cols: tl.constexpr,
        stride_t: tl.constexpr, block_size: tl.constexpr,
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
        x, cos, sin, out,
        rotary_dim: tl.constexpr, stride_t: tl.constexpr,
    ):
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


def _triton_rms_norm(x, weight, eps):
    x_c = x.contiguous()
    cols = x_c.shape[-1]
    block_size = triton.next_power_of_2(cols)
    out = torch.empty_like(x_c)
    _rms_norm_kernel[(x_c.shape[0],)](
        x_c, weight.contiguous(), out,
        eps, cols, x_c.stride(0), block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return out


def _rope_cache(positions, rotary_dim, *, base, dtype):
    device = positions.device
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) / max(half, 1))
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def _triton_rope_2d(x, positions, rope_base):
    x_c = x.contiguous()
    rotary_dim = x_c.shape[-1]
    cos, sin = _rope_cache(positions, rotary_dim, base=rope_base, dtype=x_c.dtype)
    out = torch.empty_like(x_c)
    _rope_kernel[(x_c.shape[0],)](
        x_c, cos.contiguous(), sin.contiguous(), out,
        rotary_dim, x_c.stride(0), num_warps=1,
    )
    return out


def _triton_rope_3d(x, positions, num_heads, rope_dim, rope_base):
    half = rope_dim // 2
    cos, sin = _rope_cache(positions, rope_dim, base=rope_base, dtype=x.dtype)
    cos_flat = cos.unsqueeze(1).expand(-1, num_heads, -1).reshape(-1, half).contiguous()
    sin_flat = sin.unsqueeze(1).expand(-1, num_heads, -1).reshape(-1, half).contiguous()
    x_flat = x.reshape(-1, rope_dim).contiguous()
    out = torch.empty_like(x_flat)
    _rope_kernel[(x_flat.shape[0],)](
        x_flat, cos_flat, sin_flat, out,
        rope_dim, rope_dim, num_warps=1,
    )
    return out.view_as(x)


def fused_qkv_proj_rope(
    q_c, q_norm_weight, q_b_weight,
    kv_c, kv_norm_weight, kv_b_weight,
    k_pe, positions,
    *, eps, num_heads, head_dim, nope_dim, rope_dim, v_head_dim, rope_base,
    fallback,
):
    del fallback
    tokens = q_c.shape[0]

    q_c = _triton_rms_norm(q_c, q_norm_weight, eps)
    q = F.linear(q_c, q_b_weight).view(tokens, num_heads, head_dim)
    q_nope, q_pe = q.split([nope_dim, rope_dim], dim=-1)
    q_pe = _triton_rope_3d(q_pe, positions, num_heads, rope_dim, rope_base)
    q = torch.cat((q_nope, q_pe), dim=-1)

    kv_c = _triton_rms_norm(kv_c, kv_norm_weight, eps)
    k_pe = _triton_rope_2d(k_pe, positions, rope_base)
    kv = F.linear(kv_c, kv_b_weight).view(tokens, num_heads, nope_dim + v_head_dim)
    k_nope, v = kv.split([nope_dim, v_head_dim], dim=-1)
    k = torch.cat((k_nope, k_pe.unsqueeze(1).expand(-1, num_heads, -1)), dim=-1)

    return q, k, v
