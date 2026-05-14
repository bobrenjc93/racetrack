from __future__ import annotations

import math
import os

import torch

try:
    import helion
    import helion.language as hl

    BACKEND_AVAILABLE = True
except Exception:
    helion = None
    hl = None
    BACKEND_AVAILABLE = False


def _autotune_effort() -> str:
    return os.getenv("RACETRACK_HELION_AUTOTUNE_EFFORT", "quick")


if BACKEND_AVAILABLE:

    @helion.kernel(autotune_effort=_autotune_effort())
    def _rope_kernel(
        x: torch.Tensor,
        positions: torch.Tensor,
        log_rope_base: float,
    ) -> torch.Tensor:
        out = torch.empty_like(x)
        tokens, rotary_dim = x.size()
        half = rotary_dim // 2
        for tile_t, tile_h in hl.tile([tokens, half]):
            rotary_index = tile_h.index.to(torch.float32)
            position_values = positions[tile_t].to(torch.float32).view(tile_t, 1)
            inv_freq = torch.exp(-(rotary_index / half) * log_rope_base)
            freqs = position_values * inv_freq.view(1, tile_h)
            cos = torch.cos(freqs).to(x.dtype)
            sin = torch.sin(freqs).to(x.dtype)
            x1 = x[tile_t, tile_h]
            x2 = x[tile_t, tile_h + half]
            out[tile_t, tile_h] = (x1 * cos - x2 * sin).to(x.dtype)
            out[tile_t, tile_h + half] = (x2 * cos + x1 * sin).to(x.dtype)
        return out


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
