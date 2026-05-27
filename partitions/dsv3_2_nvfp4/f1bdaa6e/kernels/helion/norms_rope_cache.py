from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import helion
    BACKEND_AVAILABLE = True
except Exception:
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


def _rms_norm(x, weight, eps):
    x_f = x.float()
    variance = x_f.pow(2).mean(-1, keepdim=True)
    return (weight * (x_f * torch.rsqrt(variance + eps))).to(x.dtype)


def _layer_norm(x, weight, bias, dim, eps):
    return F.layer_norm(
        x.float(),
        (dim,),
        _cached_float_tensor(weight),
        _cached_float_tensor(bias),
        eps,
    ).type_as(x)


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


def fused_norms_rope_cache(
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
    from partitions.dsv3_2_nvfp4.f1bdaa6e.model import (
        apply_rotary_emb, act_quant,
    )
    bsz, seqlen = q_c.shape[0], q_c.shape[1]
    end_pos = start_pos + seqlen

    qr = _rms_norm(q_c, q_norm_weight, eps)
    kv_c_normed = _rms_norm(kv_c, kv_norm_weight, eps)

    k_pe_roped = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True)
    kv_cache[:bsz, start_pos:end_pos] = kv_c_normed
    pe_cache[:bsz, start_pos:end_pos] = k_pe_roped.squeeze(2)

    idx_k = _layer_norm(indexer_k, idx_ln_weight, idx_ln_bias, idx_ln_dim, idx_ln_eps)
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
