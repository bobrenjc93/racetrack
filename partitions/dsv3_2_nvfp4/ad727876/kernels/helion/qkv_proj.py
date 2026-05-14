from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import helion
    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False


def fused_qkv_proj(
    normed_x, wq_a_weight, wkv_a_weight,
    indexer_wk_weight, indexer_wp_weight,
    *, q_lora_rank, kv_lora_rank, qk_rope_head_dim, fallback,
):
    del fallback
    qkv = F.linear(normed_x, torch.cat([wq_a_weight, wkv_a_weight], dim=0))
    q_c = qkv[..., :q_lora_rank]
    kv_c = qkv[..., q_lora_rank:q_lora_rank + kv_lora_rank]
    k_pe = qkv[..., q_lora_rank + kv_lora_rank:]
    indexer_k = F.linear(normed_x, indexer_wk_weight)
    indexer_w = F.linear(normed_x.float(), indexer_wp_weight.float())
    return q_c, kv_c, k_pe, indexer_k, indexer_w
