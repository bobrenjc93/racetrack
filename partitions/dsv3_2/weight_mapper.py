"""Map HuggingFace DeepSeek-V3.2 checkpoint weights to partition model format.

The HF checkpoint is stored in FP8 (e4m3) with block-wise scales.  This module
handles FP8 dequantization, name mapping, and parameter fusion (q_a_proj +
kv_a_proj_with_mqa -> fused_qkv_a_proj).

Usage:
    from partitions.dsv3_2.weight_mapper import load_hf_weights
    from partitions.dsv3_2.model import build_model

    model = build_model(**FULL_CONFIG)
    load_hf_weights(model, "deepseek-ai/DeepSeek-V3.2")

Known limitations:
    - Dense MLP layers (first_k_dense_replace=3 in V3.2) use intermediate_size
      (18432) which differs from moe_intermediate_size (2048). The partition
      model always uses RoutedMoE, so dense layer weights cannot be loaded
      without architecture changes.
    - HF params with no partition equivalent are silently skipped:
      self_attn.indexer.*, shared_head.*, eh_proj, per-layer embed_tokens,
      enorm, hnorm, mlp.gate.e_score_correction_bias.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

FP8_BLOCK_SIZE = 128


def dequant_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
) -> torch.Tensor:
    out_f, in_f = weight.shape
    pad_out = (block_size - out_f % block_size) % block_size
    pad_in = (block_size - in_f % block_size) % block_size
    w = weight.float()
    if pad_out > 0 or pad_in > 0:
        w = F.pad(w, (0, pad_in, 0, pad_out))
    blocks_out = w.shape[0] // block_size
    blocks_in = w.shape[1] // block_size
    w = w.view(blocks_out, block_size, blocks_in, block_size)
    s = scale_inv.float().view(blocks_out, 1, blocks_in, 1)
    result = (w * s).reshape(blocks_out * block_size, blocks_in * block_size)
    return result[:out_f, :in_f].to(torch.bfloat16)


def _get_weight(
    hf_state: dict[str, torch.Tensor],
    key: str,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    w = hf_state[key]
    scale_key = key + "_scale_inv"
    if scale_key in hf_state:
        return dequant_fp8(w, hf_state[scale_key])
    if w.dtype != dtype:
        return w.to(dtype)
    return w


SKIP_PATTERNS = [
    r"\.indexer\.",
    r"\.shared_head\.",
    r"\.eh_proj\.",
    r"\.enorm\.",
    r"\.hnorm\.",
    r"\.e_score_correction_bias$",
    r"_scale_inv$",
    r"layers\.\d+\.embed_tokens\.",
]
_SKIP_RE = re.compile("|".join(SKIP_PATTERNS))


def map_layer_attn(
    hf_state: dict[str, torch.Tensor],
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}.self_attn"
    mapped: dict[str, torch.Tensor] = {}

    q_a = _get_weight(hf_state, f"{prefix}.q_a_proj.weight")
    kv_a = _get_weight(hf_state, f"{prefix}.kv_a_proj_with_mqa.weight")
    mapped["fused_qkv_a_proj.weight"] = torch.cat([q_a, kv_a], dim=0)

    mapped["q_a_layernorm_weight"] = hf_state[f"{prefix}.q_a_layernorm.weight"]
    mapped["kv_a_layernorm_weight"] = hf_state[f"{prefix}.kv_a_layernorm.weight"]
    mapped["q_b_proj.weight"] = _get_weight(hf_state, f"{prefix}.q_b_proj.weight")
    mapped["kv_b_proj.weight"] = _get_weight(hf_state, f"{prefix}.kv_b_proj.weight")
    mapped["o_proj.weight"] = _get_weight(hf_state, f"{prefix}.o_proj.weight")
    return mapped


def map_layer_moe(
    hf_state: dict[str, torch.Tensor],
    layer_idx: int,
    n_routed_experts: int,
    n_shared_experts: int,
) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}.mlp"
    mapped: dict[str, torch.Tensor] = {}

    mapped["gate.weight"] = hf_state[f"{prefix}.gate.weight"]

    for j in range(n_routed_experts):
        ep = f"{prefix}.experts.{j}"
        mapped[f"experts.{j}.w1.weight"] = _get_weight(hf_state, f"{ep}.gate_proj.weight")
        mapped[f"experts.{j}.w3.weight"] = _get_weight(hf_state, f"{ep}.up_proj.weight")
        mapped[f"experts.{j}.w2.weight"] = _get_weight(hf_state, f"{ep}.down_proj.weight")

    if n_shared_experts > 0:
        sp = f"{prefix}.shared_experts"
        mapped["shared.w1.weight"] = _get_weight(hf_state, f"{sp}.gate_proj.weight")
        mapped["shared.w3.weight"] = _get_weight(hf_state, f"{sp}.up_proj.weight")
        mapped["shared.w2.weight"] = _get_weight(hf_state, f"{sp}.down_proj.weight")

    return mapped


def map_layer_dense_mlp(
    hf_state: dict[str, torch.Tensor],
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """Map a dense MLP layer (first_k_dense_replace layers).

    These layers use intermediate_size instead of moe_intermediate_size
    and have no experts or routing gate.  Returns weights keyed for a
    single SwiGLUExpert named 'dense_mlp'.
    """
    prefix = f"model.layers.{layer_idx}.mlp"
    return {
        "dense_mlp.w1.weight": _get_weight(hf_state, f"{prefix}.gate_proj.weight"),
        "dense_mlp.w3.weight": _get_weight(hf_state, f"{prefix}.up_proj.weight"),
        "dense_mlp.w2.weight": _get_weight(hf_state, f"{prefix}.down_proj.weight"),
    }


def build_partition_state_dict(
    hf_state: dict[str, torch.Tensor],
    *,
    num_layers: int,
    n_routed_experts: int,
    n_shared_experts: int,
    first_k_dense_replace: int = 0,
) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}

    sd["embed_tokens.weight"] = hf_state["model.embed_tokens.weight"].to(torch.bfloat16)

    for i in range(num_layers):
        lp = f"layers.{i}"
        sd[f"{lp}.attn_norm.weight"] = hf_state[f"model.layers.{i}.input_layernorm.weight"]
        sd[f"{lp}.ffn_norm.weight"] = hf_state[f"model.layers.{i}.post_attention_layernorm.weight"]

        for k, v in map_layer_attn(hf_state, i).items():
            sd[f"{lp}.attn.{k}"] = v

        if i < first_k_dense_replace:
            for k, v in map_layer_dense_mlp(hf_state, i).items():
                sd[f"{lp}.ffn.{k}"] = v
        else:
            for k, v in map_layer_moe(hf_state, i, n_routed_experts, n_shared_experts).items():
                sd[f"{lp}.ffn.{k}"] = v

    sd["norm.weight"] = hf_state["model.norm.weight"]
    sd["lm_head.weight"] = _get_weight(hf_state, "lm_head.weight")
    return sd


def load_hf_weights(
    model: torch.nn.Module,
    repo_id: str,
    *,
    num_layers: int | None = None,
    n_routed_experts: int = 256,
    n_shared_experts: int = 1,
    first_k_dense_replace: int = 0,
    hf_token: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[list[str], list[str]]:
    """Load HF checkpoint shards into a partition model.

    Returns (matched_keys, missing_keys).
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    idx_path = hf_hub_download(repo_id, "model.safetensors.index.json", token=hf_token)
    with open(idx_path) as f:
        index = json.load(f)

    if num_layers is None:
        layer_nums = set()
        for key in index["weight_map"]:
            m = re.match(r"model\.layers\.(\d+)\.", key)
            if m:
                layer_nums.add(int(m.group(1)))
        num_layers = max(layer_nums) + 1 if layer_nums else 0

    needed_shards: set[str] = set()
    for key, shard in index["weight_map"].items():
        if _SKIP_RE.search(key):
            continue
        needed_shards.add(shard)

    print(f"Loading {len(needed_shards)} checkpoint shards for {num_layers} layers ...")
    hf_state: dict[str, torch.Tensor] = {}
    cache_dir = Path(idx_path).parent
    for shard_name in sorted(needed_shards):
        shard_path = cache_dir / shard_name
        if not shard_path.exists():
            shard_path = Path(hf_hub_download(repo_id, shard_name, token=hf_token))
        with safe_open(str(shard_path), framework="pt", device=str(device)) as f:
            for key in f.keys():
                if not _SKIP_RE.search(key):
                    hf_state[key] = f.get_tensor(key)

    print("Building partition state dict ...")
    partition_sd = build_partition_state_dict(
        hf_state,
        num_layers=num_layers,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
        first_k_dense_replace=first_k_dense_replace,
    )

    model_sd = model.state_dict()
    matched = []
    missing = []
    for key in model_sd:
        if key in partition_sd:
            matched.append(key)
        else:
            missing.append(key)

    model.load_state_dict(partition_sd, strict=False)
    print(f"Loaded {len(matched)} params, {len(missing)} missing in checkpoint")
    return matched, missing
