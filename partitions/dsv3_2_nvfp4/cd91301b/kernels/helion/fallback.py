from __future__ import annotations

import torch
import torch.distributed as dist

try:
    import helion  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False


def fused_full_topk_indexer(
    indexer,
    x: torch.Tensor,
    qr: torch.Tensor,
    start_pos: int,
    freqs_cis: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    fallback,
) -> torch.Tensor:
    bsz, seqlen, _ = x.size()
    end_pos = start_pos + seqlen
    if start_pos == 0:
        indexer._racetrack_pending_full_topk = []
    elif not hasattr(indexer, "_racetrack_pending_full_topk"):
        indexer._racetrack_pending_full_topk = []
    if end_pos > indexer.index_topk:
        _flush_pending_full_topk(indexer)
        return fallback(indexer, x, qr, start_pos, freqs_cis, mask)

    indexer._racetrack_pending_full_topk.append((start_pos, x, freqs_cis))
    return torch.arange(
        end_pos,
        device=x.device,
        dtype=torch.long,
    ).view(1, 1, end_pos).expand(bsz, seqlen, end_pos).contiguous()


def _flush_pending_full_topk(indexer) -> None:
    pending = getattr(indexer, "_racetrack_pending_full_topk", [])
    if not pending:
        return
    for start_pos, x, freqs_cis in pending:
        _write_indexer_k_cache(indexer, x, start_pos, freqs_cis)
    pending.clear()


def _write_indexer_k_cache(
    indexer,
    x: torch.Tensor,
    start_pos: int,
    freqs_cis: torch.Tensor,
) -> None:
    bsz, seqlen, _ = x.size()
    end_pos = start_pos + seqlen

    from inference import model as real_model

    k = indexer.wk(x)
    k = indexer.k_norm(k)
    k_pe, k_nope = torch.split(
        k,
        [indexer.rope_head_dim, indexer.head_dim - indexer.rope_head_dim],
        dim=-1,
    )
    k_pe = real_model.apply_rotary_emb(
        k_pe.unsqueeze(2),
        freqs_cis,
        False,
    ).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)
    k = real_model.rotate_activation(k)
    k_fp8, k_scale = real_model.act_quant(
        k,
        real_model.block_size,
        indexer.scale_fmt,
    )
    indexer.k_cache[:bsz, start_pos:end_pos] = k_fp8
    indexer.k_scale_cache[:bsz, start_pos:end_pos] = k_scale


def fused_single_token_moe(
    moe,
    x: torch.Tensor,
    *,
    fallback,
) -> torch.Tensor:
    shape = x.size()
    flat = x.view(-1, moe.dim)
    if flat.shape[0] != 1:
        return fallback(moe, x)

    from inference import model as real_model

    weights, indices = moe.gate(flat)
    y = torch.zeros_like(flat, dtype=torch.float32)
    for top, expert_id in enumerate(indices[0].tolist()):
        if moe.experts_start_idx <= expert_id < moe.experts_end_idx:
            y += moe.experts[expert_id](flat) * weights[0, top, None]
    y += moe.shared_experts(flat)
    if real_model.world_size > 1:
        dist.all_reduce(y)
    return y.type_as(flat).view(shape)


def fused_qkv_proj(
    normed_x,
    wq_a_weight,
    wkv_a_weight,
    indexer_wk_weight,
    indexer_wp_weight,
    *,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_head_dim,
    fallback,
):
    return fallback(
        normed_x, wq_a_weight, wkv_a_weight,
        indexer_wk_weight, indexer_wp_weight,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )


def fused_attn_norm_qkv_prefill(
    x,
    residual,
    norm_weight,
    wq_a_weight,
    wkv_a_weight,
    *,
    eps,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_head_dim,
    fallback,
):
    return fallback(
        x, residual, norm_weight, wq_a_weight, wkv_a_weight,
        eps=eps,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )


def fused_norms_rope_cache_prefill(
    q_c,
    q_norm_weight,
    kv_c,
    kv_norm_weight,
    k_pe,
    freqs_cis,
    kv_cache,
    pe_cache,
    *,
    eps,
    start_pos,
    fallback,
):
    return fallback(
        q_c, q_norm_weight, kv_c, kv_norm_weight, k_pe,
        freqs_cis, kv_cache, pe_cache,
        eps=eps, start_pos=start_pos,
    )


def fused_prefill_qkv_b_rope_cache(
    q_c,
    q_norm_weight,
    wq_b_weight,
    kv_c,
    kv_norm_weight,
    wkv_b_weight,
    k_pe,
    freqs_cis,
    kv_cache,
    pe_cache,
    *,
    eps,
    n_heads,
    qk_head_dim,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    start_pos,
    fallback,
):
    return fallback(
        q_c, q_norm_weight, wq_b_weight,
        kv_c, kv_norm_weight, wkv_b_weight,
        k_pe, freqs_cis, kv_cache, pe_cache,
        eps=eps,
        n_heads=n_heads,
        qk_head_dim=qk_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        start_pos=start_pos,
    )


def fused_norms_rope_cache(
    q_c,
    q_norm_weight,
    kv_c,
    kv_norm_weight,
    k_pe,
    indexer_k,
    idx_ln_weight,
    idx_ln_bias,
    freqs_cis,
    H,
    kv_cache,
    pe_cache,
    idx_k_cache,
    idx_k_scale_cache,
    *,
    eps,
    idx_ln_dim,
    idx_ln_eps,
    rope_head_dim,
    start_pos,
    block_size,
    fallback,
):
    return fallback(
        q_c, q_norm_weight, kv_c, kv_norm_weight, k_pe, indexer_k,
        idx_ln_weight, idx_ln_bias, freqs_cis, H,
        kv_cache, pe_cache, idx_k_cache, idx_k_scale_cache,
        eps=eps,
        idx_ln_dim=idx_ln_dim,
        idx_ln_eps=idx_ln_eps,
        rope_head_dim=rope_head_dim,
        start_pos=start_pos,
        block_size=block_size,
    )


def fused_q_prefill_proj(
    qr,
    wq_b_weight,
    *,
    n_heads,
    qk_head_dim,
    qk_nope_head_dim,
    fallback,
):
    return fallback(
        qr, wq_b_weight,
        n_heads=n_heads,
        qk_head_dim=qk_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
    )


def fused_q_indexer_proj(
    qr,
    wq_b_weight,
    idx_wq_b_weight,
    wkv_b_weight,
    *,
    n_heads,
    qk_head_dim,
    qk_nope_head_dim,
    kv_lora_rank,
    idx_n_heads,
    idx_head_dim,
    fallback,
):
    return fallback(
        qr, wq_b_weight, idx_wq_b_weight, wkv_b_weight,
        n_heads=n_heads,
        qk_head_dim=qk_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        idx_n_heads=idx_n_heads,
        idx_head_dim=idx_head_dim,
    )


def fused_q_rope_prefill(q_pe, freqs_cis, *, fallback):
    return fallback(q_pe, freqs_cis)


def fused_q_rope_indexer_score(
    q_pe,
    idx_q,
    indexer_w,
    freqs_cis,
    H,
    idx_k_cache,
    idx_k_scale_cache,
    *,
    rope_head_dim,
    idx_n_heads,
    idx_softmax_scale,
    end_pos,
    block_size,
    fallback,
):
    return fallback(
        q_pe, idx_q, indexer_w, freqs_cis, H,
        idx_k_cache, idx_k_scale_cache,
        rope_head_dim=rope_head_dim,
        idx_n_heads=idx_n_heads,
        idx_softmax_scale=idx_softmax_scale,
        end_pos=end_pos,
        block_size=block_size,
    )
