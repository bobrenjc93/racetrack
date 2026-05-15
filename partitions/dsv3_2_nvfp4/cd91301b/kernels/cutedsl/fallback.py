from __future__ import annotations

try:
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False


def fused_residual_norm(
    x,
    residual,
    weight,
    *,
    eps,
    fallback,
):
    return fallback(x, residual, weight, eps=eps)


def fused_swiglu(gate, up, *, fallback):
    return fallback(gate, up)


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
