from __future__ import annotations

import torch

try:
    import cutlass
    import cutlass.cute as cute
    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False


def fused_q_rope_indexer_score(
    q_pe, idx_q, indexer_w,
    freqs_cis, H,
    idx_k_cache, idx_k_scale_cache,
    *, rope_head_dim, idx_n_heads, idx_softmax_scale,
    end_pos, block_size,
    fallback,
):
    del fallback
    from partitions.dsv3_2_nvfp4.ad727876.model import (
        apply_rotary_emb, hadamard_transform, act_quant, fp8_index,
    )
    bsz, seqlen = q_pe.shape[0], q_pe.shape[1]

    q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=True)

    idx_q_pe, idx_q_nope = torch.split(
        idx_q, [rope_head_dim, idx_q.shape[-1] - rope_head_dim], dim=-1,
    )
    idx_q_pe = apply_rotary_emb(idx_q_pe, freqs_cis, interleaved=False)
    idx_q = torch.cat([idx_q_pe, idx_q_nope], dim=-1)
    idx_q = hadamard_transform(idx_q, H)
    idx_q_fp8, idx_q_scale = act_quant(idx_q, block_size)

    weights = indexer_w * idx_n_heads ** -0.5
    weights = (weights.unsqueeze(-1) * idx_q_scale * idx_softmax_scale).squeeze(-1)

    k_s = idx_k_scale_cache[:bsz, :end_pos].squeeze(-1).contiguous()
    k_cached = idx_k_cache[:bsz, :end_pos].contiguous()
    index_score = fp8_index(idx_q_fp8.float(), weights, k_cached, k_s)
    topk_indices = index_score.topk(min(end_pos, seqlen * 2), dim=-1)[1]

    return q_pe, topk_indices
