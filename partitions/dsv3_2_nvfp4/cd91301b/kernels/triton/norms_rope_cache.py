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


_FLOAT_TENSOR_CACHE = {}


def _cached_float_tensor(tensor):
    if tensor.dtype == torch.float32:
        return tensor
    if torch.is_grad_enabled():
        return tensor.float()
    key = (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        getattr(tensor, "_version", 0),
    )
    cached = _FLOAT_TENSOR_CACHE.get(key)
    if cached is None:
        cached = tensor.float().contiguous()
        _FLOAT_TENSOR_CACHE[key] = cached
    return cached


def _hadamard_transform(x, H):
    d = x.shape[-1]
    return (x.float() @ _cached_float_tensor(H)[:d, :d] * (d ** -0.5)).type_as(x)


def _act_quant_unit_scale(x, block_size, act_quant):
    if x.size(-1) % block_size == 0:
        return act_quant(x, block_size)
    try:
        return x.contiguous().to(torch.float8_e4m3fn), None
    except RuntimeError:
        return x.float(), None


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
    def _layer_norm_kernel(
        x, weight, bias, out,
        eps: tl.constexpr, cols: tl.constexpr,
        stride_t: tl.constexpr, block_size: tl.constexpr,
    ):
        token = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < cols
        values = tl.load(x + token * stride_t + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(values, axis=0) / cols
        centered = values - mean
        variance = tl.sum(centered * centered, axis=0) / cols
        scale = tl.rsqrt(variance + eps)
        w = tl.load(weight + offsets, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out + token * cols + offsets, centered * scale * w + b, mask=mask)

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

    @triton.jit
    def _act_quant_kernel(
        x, out, scale_out,
        n_elements: tl.constexpr,
        block_size_q: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = tl.arange(0, block_size_q)
        base = pid * block_size_q
        mask = (base + offsets) < n_elements
        values = tl.load(x + base + offsets, mask=mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(values), axis=0)
        amax = tl.where(amax < 1e-4, 1e-4, amax)
        s = amax / 448.0
        scaled = values / s
        scaled = tl.where(scaled > 448.0, 448.0, scaled)
        scaled = tl.where(scaled < -448.0, -448.0, scaled)
        tl.store(out + base + offsets, scaled, mask=mask)
        tl.store(scale_out + pid, s)


def _triton_rms_norm(x, weight, eps):
    x_c = x.contiguous()
    cols = x_c.shape[-1]
    block_size = triton.next_power_of_2(cols)
    rows = x_c.shape[0] if x_c.dim() == 2 else x_c.shape[0] * x_c.shape[1]
    x_flat = x_c.reshape(rows, cols)
    out = torch.empty_like(x_flat)
    _rms_norm_kernel[(rows,)](
        x_flat, weight.contiguous(), out,
        eps, cols, cols, block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return out.view_as(x_c)


def _triton_layer_norm(x, weight, bias, dim, eps):
    x_c = x.contiguous()
    rows = x_c.shape[0] if x_c.dim() == 2 else x_c.shape[0] * x_c.shape[1]
    x_flat = x_c.reshape(rows, dim)
    block_size = triton.next_power_of_2(dim)
    out = torch.empty_like(x_flat)
    _layer_norm_kernel[(rows,)](
        x_flat, weight.contiguous(), bias.contiguous(), out,
        eps, dim, dim, block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return out.view_as(x_c)


def _rope_cache(positions, rotary_dim, *, base, dtype):
    device = positions.device
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) / max(half, 1))
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def fused_indexer_k_path(
    q_c, q_norm_weight,
    kv_c, kv_norm_weight,
    k_pe, indexer_k,
    idx_ln_weight, idx_ln_bias,
    freqs_cis, H,
    kv_cache, pe_cache,
    idx_k_cache, idx_k_scale_cache,
    *, eps, idx_ln_dim, idx_ln_eps, rope_head_dim,
    start_pos, block_size,
    fallback,
):
    del fallback
    bsz, seqlen = q_c.shape[0], q_c.shape[1]
    end_pos = start_pos + seqlen

    qr = _triton_rms_norm(q_c, q_norm_weight, eps)
    kv_c_normed = _triton_rms_norm(kv_c, kv_norm_weight, eps)

    k_pe_roped = fallback.__func__(
        q_c, q_norm_weight, kv_c, kv_norm_weight, k_pe, indexer_k,
        idx_ln_weight, idx_ln_bias, freqs_cis, H,
        kv_cache, pe_cache, idx_k_cache, idx_k_scale_cache,
        eps=eps, idx_ln_dim=idx_ln_dim, idx_ln_eps=idx_ln_eps,
        rope_head_dim=rope_head_dim, start_pos=start_pos, block_size=block_size,
    ) if False else None

    from partitions.dsv3_2_nvfp4.cd91301b.model import (
        apply_rotary_emb, act_quant,
    )

    k_pe_roped = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True)
    kv_cache[:bsz, start_pos:end_pos] = kv_c_normed
    pe_cache[:bsz, start_pos:end_pos] = k_pe_roped.squeeze(2)

    idx_k = _triton_layer_norm(indexer_k, idx_ln_weight, idx_ln_bias, idx_ln_dim, idx_ln_eps)
    idx_k_pe, idx_k_nope = torch.split(
        idx_k, [rope_head_dim, idx_k.shape[-1] - rope_head_dim], dim=-1,
    )
    idx_k_pe = apply_rotary_emb(idx_k_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2)
    idx_k = torch.cat([idx_k_pe, idx_k_nope], dim=-1)
    idx_k = _hadamard_transform(idx_k, H)
    idx_k_fp8, idx_k_scale = _act_quant_unit_scale(idx_k, block_size, act_quant)
    idx_k_cache[:bsz, start_pos:end_pos] = idx_k_fp8.float()
    if idx_k_scale is None:
        idx_k_scale_cache[:bsz, start_pos:end_pos] = 1.0
    else:
        idx_k_scale_cache[:bsz, start_pos:end_pos] = idx_k_scale

    return qr
