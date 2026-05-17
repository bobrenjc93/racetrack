PARTITION_NOTES = '4-kernel NVFP4 fusion: AR+RMS+QKV, norms+RoPE+cache, Q/Indexer scoring, Q RoPE+quant'

GRAPH_NODES = {
    "fused_ar_rms_qkv_proj": ["qkv_a_proj", "indexer_k_proj", "indexer_w"],
    "fused_indexer_k_path": ["kv_c_rms", "kv_rope", "kv_quant_fp8", "mla_cache", "q_rms", "indexer_ln", "indexer_rope", "indexer_quant_fp8", "indexer_cache"],
    "fused_q_indexer_score": [
        "q_b_proj", "indexer_q_proj", "w_uk_t",
    ],
    "fused_q_rope_quant": ["q_rope", "cat_q", "q_quant_fp8", "indexer_q_rope", "indexer_q_fp8", "indexer_w_scale"],
}

FUSED_OPS = [
    {"name": "fused_ar_rms_qkv_proj", "kind": "module_patch"},
    {"name": "fused_indexer_k_path", "kind": "module_patch"},
    {"name": "fused_q_indexer_score", "kind": "module_patch"},
    {"name": "fused_q_rope_quant", "kind": "module_patch"},
]
