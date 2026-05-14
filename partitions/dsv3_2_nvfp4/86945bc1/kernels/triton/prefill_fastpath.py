from __future__ import annotations

import torch
import torch.nn.functional as F
import importlib

try:
    import triton
    import triton.language as tl

    BACKEND_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    BACKEND_AVAILABLE = False


_CAT_CACHE = {}


def _cached_cat_weights(*weights):
    if torch.is_grad_enabled():
        return torch.cat(weights, dim=0)
    key = tuple(
        (
            weight.data_ptr(),
            tuple(weight.shape),
            tuple(weight.stride()),
            str(weight.dtype),
            str(weight.device),
            getattr(weight, "_version", 0),
        )
        for weight in weights
    )
    cached = _CAT_CACHE.get(key)
    if cached is None:
        cached = torch.cat(weights, dim=0).contiguous()
        _CAT_CACHE[key] = cached
    return cached


if BACKEND_AVAILABLE:

    @triton.jit
    def _rms_norm_kernel(
        x, weight, out,
        eps: tl.constexpr, cols: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < cols
        values = tl.load(x + row * cols + offsets, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / cols
        scale = tl.rsqrt(variance + eps)
        tl.store(out + row * cols + offsets, values * scale * weights, mask=mask)

    @triton.jit
    def _residual_rms_norm_kernel(
        x, residual, weight, hidden_out, normed_out,
        eps: tl.constexpr, cols: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < cols
        x_values = tl.load(x + row * cols + offsets, mask=mask, other=0.0).to(tl.float32)
        r_values = tl.load(residual + row * cols + offsets, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        hidden = x_values + r_values
        tl.store(hidden_out + row * cols + offsets, hidden, mask=mask)
        variance = tl.sum(hidden * hidden, axis=0) / cols
        scale = tl.rsqrt(variance + eps)
        tl.store(normed_out + row * cols + offsets, hidden * scale * weights, mask=mask)


def _triton_rms_norm(x, weight, eps):
    x_c = x.contiguous()
    shape = x_c.shape
    cols = shape[-1]
    rows = x_c.numel() // cols
    x_flat = x_c.view(rows, cols)
    out = torch.empty_like(x_flat)
    block_size = triton.next_power_of_2(cols)
    _rms_norm_kernel[(rows,)](
        x_flat, weight.contiguous(), out,
        eps, cols, block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return out.view(shape)


def _triton_residual_rms_norm(x, residual, weight, eps):
    x_c = x.contiguous()
    residual_c = residual.contiguous()
    shape = x_c.shape
    cols = shape[-1]
    rows = x_c.numel() // cols
    x_flat = x_c.view(rows, cols)
    residual_flat = residual_c.view(rows, cols)
    hidden = torch.empty_like(x_flat)
    normed = torch.empty_like(x_flat)
    block_size = triton.next_power_of_2(cols)
    _residual_rms_norm_kernel[(rows,)](
        x_flat, residual_flat, weight.contiguous(),
        hidden, normed,
        eps, cols, block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return hidden.view(shape), normed.view(shape)


def fused_attn_norm_qkv_prefill(
    x, residual, norm_weight, wq_a_weight, wkv_a_weight,
    *, eps, q_lora_rank, kv_lora_rank, qk_rope_head_dim, fallback,
):
    del fallback, qk_rope_head_dim
    if residual is None:
        residual_out = x
        normed_x = _triton_rms_norm(x, norm_weight, eps)
    else:
        residual_out, normed_x = _triton_residual_rms_norm(x, residual, norm_weight, eps)
    qkv = F.linear(normed_x, _cached_cat_weights(wq_a_weight, wkv_a_weight))
    q_c = qkv[..., :q_lora_rank]
    kv_c = qkv[..., q_lora_rank:q_lora_rank + kv_lora_rank]
    k_pe = qkv[..., q_lora_rank + kv_lora_rank:]
    return residual_out, q_c, kv_c, k_pe


def fused_norms_rope_cache_prefill(
    q_c, q_norm_weight,
    kv_c, kv_norm_weight,
    k_pe, freqs_cis,
    kv_cache, pe_cache,
    *, eps, start_pos, fallback,
):
    del fallback
    apply_rotary_emb = importlib.import_module(
        "partitions.dsv3_2_nvfp4.86945bc1.model"
    ).apply_rotary_emb

    bsz, seqlen = q_c.shape[0], q_c.shape[1]
    end_pos = start_pos + seqlen
    qr = _triton_rms_norm(q_c, q_norm_weight, eps)
    kv_c_normed = _triton_rms_norm(kv_c, kv_norm_weight, eps)
    k_pe_roped = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True)
    kv_cache[:bsz, start_pos:end_pos] = kv_c_normed
    pe_cache[:bsz, start_pos:end_pos] = k_pe_roped.squeeze(2)
    return qr


def fused_q_prefill_proj(
    qr, wq_b_weight,
    *, n_heads, qk_head_dim, qk_nope_head_dim, fallback,
):
    del fallback
    bsz, seqlen, _ = qr.shape
    q = F.linear(qr, wq_b_weight).view(bsz, seqlen, n_heads, qk_head_dim)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_head_dim - qk_nope_head_dim], dim=-1)
    return q_nope, q_pe


def fused_q_rope_prefill(q_pe, freqs_cis, *, fallback):
    del fallback
    apply_rotary_emb = importlib.import_module(
        "partitions.dsv3_2_nvfp4.86945bc1.model"
    ).apply_rotary_emb

    return apply_rotary_emb(q_pe, freqs_cis, interleaved=True)
