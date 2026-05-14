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


def fused_q_indexer_proj(
    qr, wq_b_weight, idx_wq_b_weight, wkv_b_weight,
    *, n_heads, qk_head_dim, qk_nope_head_dim,
    kv_lora_rank, idx_n_heads, idx_head_dim,
    fallback,
):
    del fallback
    bsz, seqlen, _ = qr.shape

    q_out_dim = n_heads * qk_head_dim
    q_idx = F.linear(qr, _cached_cat_weights(wq_b_weight, idx_wq_b_weight))
    q = q_idx[..., :q_out_dim].view(bsz, seqlen, n_heads, qk_head_dim)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_head_dim - qk_nope_head_dim], dim=-1)

    wkv_b = wkv_b_weight.view(n_heads, -1, kv_lora_rank)
    q_nope_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope_head_dim])

    idx_q = q_idx[..., q_out_dim:].view(bsz, seqlen, idx_n_heads, idx_head_dim)

    return q_nope, q_nope_absorbed, q_pe, idx_q
