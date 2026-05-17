PARTITION_NOTES = 'Prefill + residual_norm + swiglu + indexer shortcircuit + single-token MoE'

GRAPH_NODES = {
    "fused_ar_rms_qkv_proj": ["qkv_a_proj", "indexer_k_proj", "indexer_w"],
    "fused_attn_norm_qkv_prefill": ["attn_norm", "qkv_a_proj"],
    "fused_norms_rope_cache_prefill": ["q_rms", "kv_c_rms", "kv_rope", "mla_cache"],
    "fused_q_prefill_proj": ["q_b_proj"],
    "fused_q_rope_prefill": ["q_rope"],
    "fused_full_topk_indexer": ["indexer_mqa", "logits_topk", "topk_page_idx"],
    "fused_single_token_moe": ["gate_router", "topk_softmax", "expert_sum"],
    "fused_residual_norm": ["attn_residual_add", "ffn_rms"],
    "fused_swiglu": ["ffn_silu", "ffn_gate_up_mul"],
    "fused_indexer_k_path": ["kv_c_rms", "kv_rope", "kv_quant_fp8", "mla_cache", "q_rms", "indexer_ln", "indexer_rope", "indexer_quant_fp8", "indexer_cache"],
    "fused_q_indexer_score": [
        "q_b_proj", "indexer_q_proj", "w_uk_t",
    ],
    "fused_q_rope_quant": ["q_rope", "cat_q", "q_quant_fp8", "indexer_q_rope", "indexer_q_fp8", "indexer_w_scale"],
}

FUSED_OPS = [
    {"name": "fused_swiglu", "kind": "fx_pattern"},
    {"name": "fused_residual_norm", "kind": "fx_pattern"},
    {"name": "fused_full_topk_indexer", "kind": "pre_trace"},
    {"name": "fused_single_token_moe", "kind": "pre_trace"},
    {"name": "fused_ar_rms_qkv_proj", "kind": "module_patch"},
    {"name": "fused_indexer_k_path", "kind": "module_patch"},
    {"name": "fused_q_indexer_score", "kind": "module_patch"},
    {"name": "fused_q_rope_quant", "kind": "module_patch"},
    {"name": "fused_attn_norm_qkv_prefill", "kind": "module_patch"},
    {"name": "fused_norms_rope_cache_prefill", "kind": "module_patch"},
    {"name": "fused_q_prefill_proj", "kind": "module_patch"},
    {"name": "fused_q_rope_prefill", "kind": "module_patch"},
]
