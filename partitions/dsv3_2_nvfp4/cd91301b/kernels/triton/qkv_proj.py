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
_FLOAT_WEIGHT_CACHE = {}


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


def _cached_float_weight(weight):
    if weight.dtype == torch.float32:
        return weight
    if torch.is_grad_enabled():
        return weight.float()
    key = (
        weight.data_ptr(),
        tuple(weight.shape),
        tuple(weight.stride()),
        str(weight.dtype),
        str(weight.device),
        getattr(weight, "_version", 0),
    )
    cached = _FLOAT_WEIGHT_CACHE.get(key)
    if cached is None:
        cached = weight.float().contiguous()
        _FLOAT_WEIGHT_CACHE[key] = cached
    return cached


def fused_qkv_proj(
    normed_x, wq_a_weight, wkv_a_weight,
    indexer_wk_weight, indexer_wp_weight,
    *, q_lora_rank, kv_lora_rank, qk_rope_head_dim, fallback,
):
    del fallback
    qkv = F.linear(normed_x, _cached_cat_weights(wq_a_weight, wkv_a_weight))
    q_c = qkv[..., :q_lora_rank]
    kv_c = qkv[..., q_lora_rank:q_lora_rank + kv_lora_rank]
    k_pe = qkv[..., q_lora_rank + kv_lora_rank:]
    indexer_k = F.linear(normed_x, indexer_wk_weight)
    indexer_w = F.linear(normed_x.float(), _cached_float_weight(indexer_wp_weight))
    return q_c, kv_c, k_pe, indexer_k, indexer_w
