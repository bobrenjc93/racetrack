from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
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


def _rms_norm(x, weight, eps):
    x_f = x.float()
    variance = x_f.pow(2).mean(-1, keepdim=True)
    return (weight * (x_f * torch.rsqrt(variance + eps))).to(x.dtype)


def _residual_rms_norm(x, residual, weight, eps):
    hidden = (x.float() + residual.float()).to(x.dtype)
    normed = _rms_norm(hidden, weight, eps)
    return hidden, normed


def fused_attn_norm_qkv_prefill(
    x, residual, norm_weight, wq_a_weight, wkv_a_weight,
    *, eps, q_lora_rank, kv_lora_rank, qk_rope_head_dim, fallback,
):
    del fallback, qk_rope_head_dim
    if residual is None:
        residual_out = x
        normed_x = _rms_norm(x, norm_weight, eps)
    else:
        residual_out, normed_x = _residual_rms_norm(x, residual, norm_weight, eps)
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
    from partitions.dsv3_2_nvfp4.cd91301b.model import apply_rotary_emb

    bsz, seqlen = q_c.shape[0], q_c.shape[1]
    end_pos = start_pos + seqlen
    qr = _rms_norm(q_c, q_norm_weight, eps)
    kv_c_normed = _rms_norm(kv_c, kv_norm_weight, eps)
    k_pe_roped = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True)
    kv_cache[:bsz, start_pos:end_pos] = kv_c_normed
    pe_cache[:bsz, start_pos:end_pos] = k_pe_roped.squeeze(2)
    return qr


def fused_prefill_qkv_b_rope_cache(
    q_c, q_norm_weight, wq_b_weight,
    kv_c, kv_norm_weight, wkv_b_weight,
    k_pe, freqs_cis,
    kv_cache, pe_cache,
    *, eps, n_heads, qk_head_dim, qk_nope_head_dim,
    qk_rope_head_dim, v_head_dim, start_pos, fallback,
):
    del fallback
    from partitions.dsv3_2_nvfp4.cd91301b.model import apply_rotary_emb

    bsz, seqlen = q_c.shape[0], q_c.shape[1]
    end_pos = start_pos + seqlen

    qr = _rms_norm(q_c, q_norm_weight, eps)
    q = F.linear(qr, wq_b_weight).view(bsz, seqlen, n_heads, qk_head_dim)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=True)

    kv_c_normed = _rms_norm(kv_c, kv_norm_weight, eps)
    kv_cache[:bsz, start_pos:end_pos] = kv_c_normed
    k_pe_roped = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True).squeeze(2)
    pe_cache[:bsz, start_pos:end_pos] = k_pe_roped

    kv_expanded = F.linear(kv_c_normed, wkv_b_weight).view(
        bsz, seqlen, n_heads, qk_nope_head_dim + v_head_dim,
    )
    return q_nope, q_pe, kv_expanded, k_pe_roped


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
    from partitions.dsv3_2_nvfp4.cd91301b.model import apply_rotary_emb

    return apply_rotary_emb(q_pe, freqs_cis, interleaved=True)
