PARTITION_NOTES = 'Deep-fusion with SDPA: attn_norm_qkv + qkv_proj_rope + residual_norm + single_token_moe'

GRAPH_NODES = {
    "fused_attn_norm_qkv": ["attn_norm", "qkv_proj", "split_qkv"],
    "fused_qkv_proj_rope": [
        "rms_norm_q", "q_b_proj", "rope_q", "cat_q",
        "rms_norm_kv", "rope_kpe", "kv_b_proj", "split_kv", "cat_k",
    ],
    "fused_single_token_moe": ["gate_router", "topk_softmax", "expert_sum"],
    "fused_residual_norm": ["res_add_attn", "ffn_norm"],
}

FUSED_OPS = [
    {"name": "fused_residual_norm", "kind": "fx_pattern"},
    {"name": "fused_single_token_moe", "kind": "pre_trace"},
    {"name": "fused_attn_norm_qkv", "kind": "module_patch"},
    {"name": "fused_qkv_proj_rope", "kind": "module_patch"},
]
