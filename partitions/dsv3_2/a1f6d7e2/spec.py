PARTITION_NOTES = 'Fuses q/kv RMSNorm and RoPE at the MLA/indexer boundary.'

GRAPH_NODES = {
    "fused_norm_rope": ["rms_norm_q", "rms_norm_kv", "rope_kpe"],
}

FUSED_OPS = [
    {"name": "fused_norm_rope", "kind": "fx_pattern"},
]
