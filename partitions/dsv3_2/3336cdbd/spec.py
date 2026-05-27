PARTITION_NOTES = 'Multi-kernel: residual_norm + swiglu + act_quant + indexer shortcircuit + MoE + gate-up proj'

GRAPH_NODES = {
    "fused_norm_rope": ["rms_norm_q", "rms_norm_kv", "rope_kpe"],
    "fused_full_topk_indexer": ["indexer_q_proj", "indexer_score", "indexer_topk"],
    "fused_single_token_moe": ["gate_router", "topk_softmax", "expert_sum"],
    "fused_mlp_gate_up_proj": ["w1_proj", "w3_proj"],
    "fused_residual_norm": ["res_add_attn", "ffn_norm"],
    "fused_swiglu": ["swiglu"],
}

FUSED_OPS = [
    {"name": "fused_swiglu", "kind": "fx_pattern"},
    {"name": "fused_residual_norm", "kind": "fx_pattern"},
    {"name": "fused_act_quant", "kind": "fx_pattern"},
    {"name": "fused_full_topk_indexer", "kind": "pre_trace"},
    {"name": "fused_single_token_moe", "kind": "pre_trace"},
    {"name": "fused_mlp_gate_up_proj", "kind": "module_patch"},
]
