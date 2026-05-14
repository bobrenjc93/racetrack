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


def fused_q_indexer_proj(
    qr, wq_b_weight, idx_wq_b_weight, wkv_b_weight,
    *, n_heads, qk_head_dim, qk_nope_head_dim,
    kv_lora_rank, idx_n_heads, idx_head_dim,
    fallback,
):
    del fallback
    bsz, seqlen, _ = qr.shape

    q = F.linear(qr, wq_b_weight).view(bsz, seqlen, n_heads, qk_head_dim)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_head_dim - qk_nope_head_dim], dim=-1)

    wkv_b = wkv_b_weight.view(n_heads, -1, kv_lora_rank)
    q_nope_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope_head_dim])

    idx_q = F.linear(qr, idx_wq_b_weight).view(bsz, seqlen, idx_n_heads, idx_head_dim)

    return q_nope, q_nope_absorbed, q_pe, idx_q
