from __future__ import annotations

import torch

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


def _rope_cache(positions, rotary_dim, *, base, dtype):
    device = positions.device
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) / max(half, 1))
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def fused_norm_rope(
    q_c, q_weight, kv_c, kv_weight, k_pe, positions,
    *, eps, rope_base, fallback,
):
    del fallback

    def _norm(x, w):
        cols = x.shape[-1]
        block_size = triton.next_power_of_2(cols)
        out = torch.empty((x.shape[0], cols), device=x.device, dtype=x.dtype)
        _rms_norm_kernel[(x.shape[0],)](
            x, w.contiguous(), out, eps, cols, x.stride(0), block_size,
            num_warps=8 if block_size >= 2048 else 4,
        )
        return out

    def _rope(x):
        rotary_dim = x.shape[-1]
        cos, sin = _rope_cache(positions, rotary_dim, base=rope_base, dtype=x.dtype)
        out = torch.empty_like(x)
        _rope_kernel[(x.shape[0],)](
            x, cos.contiguous(), sin.contiguous(), out,
            rotary_dim, x.stride(0), num_warps=1,
        )
        return out

    return _norm(q_c, q_weight), _norm(kv_c, kv_weight), _rope(k_pe.contiguous())
