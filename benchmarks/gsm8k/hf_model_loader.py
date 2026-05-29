"""Load Hugging Face DeepSeek-V3.2 shards into the model-parallel runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

SLICE = Literal["full", "row", "col"]
POST_LOAD_TRANSFORM_HOOKS = ("rebuild_derived_weights", "fuse_indexer_weights")
FP8_BLOCK_SIZE = 128


@dataclass(frozen=True)
class Assignment:
    source_key: str
    target: torch.Tensor
    slice_kind: SLICE
    block_elems: int | None = None


def load_hf_sharded_weights(
    model: torch.nn.Module,
    *,
    repo_id: str,
    hf_token: str,
    rank: int,
    world_size: int,
) -> int:
    """Load HF-format shards into ``racetrack.models.deepseek.Transformer``.

    The HF checkpoint stores full tensors. The inference model is tensor
    parallel, so this loader slices column-parallel rows, row-parallel columns,
    vocab shards, and local MoE experts for the current rank.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index_path = Path(
        hf_hub_download(repo_id, "model.safetensors.index.json", token=hf_token)
    )
    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]

    assignments = _build_assignments(model, rank=rank, world_size=world_size)
    by_shard: dict[str, list[Assignment]] = {}
    for assignment in assignments:
        if assignment.source_key not in weight_map:
            raise KeyError(f"HF checkpoint is missing {assignment.source_key}")
        by_shard.setdefault(weight_map[assignment.source_key], []).append(assignment)

    loaded = 0
    for shard_name in sorted(by_shard):
        shard_path = Path(hf_hub_download(repo_id, shard_name, token=hf_token))
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for assignment in by_shard[shard_name]:
                value = shard.get_tensor(assignment.source_key)
                value = _slice_for_rank(value, assignment, rank)
                _copy_into(assignment.target, value)
                loaded += 1
    return loaded


def run_post_load_transforms(
    model: torch.nn.Module,
    *,
    hook_names: tuple[str, ...] = POST_LOAD_TRANSFORM_HOOKS,
) -> list[str]:
    """Run module hooks that rebuild derived tensors after checkpoint load.

    A module may expose methods such as ``rebuild_derived_weights`` or
    ``fuse_indexer_weights`` when some inference-time tensors are derived from
    checkpoint parameters rather than stored directly in the checkpoint.
    """
    called: list[str] = []
    with torch.no_grad():
        for module_name, module in model.named_modules():
            for hook_name in hook_names:
                hook = getattr(module, hook_name, None)
                if callable(hook):
                    hook()
                    label = module_name or module.__class__.__name__
                    called.append(f"{label}.{hook_name}")
                    break
    return called


def _copy_into(target: torch.Tensor, value: torch.Tensor) -> None:
    if tuple(value.shape) != tuple(target.shape):
        raise ValueError(
            f"Shape mismatch for target {tuple(target.shape)} from {tuple(value.shape)}"
        )
    with torch.no_grad():
        target.copy_(value.to(device=target.device, dtype=target.dtype))


def _slice_for_rank(
    value: torch.Tensor,
    assignment: Assignment,
    rank: int,
) -> torch.Tensor:
    slice_kind = assignment.slice_kind
    target = assignment.target
    if slice_kind == "full":
        return value
    dim = 0 if slice_kind == "row" else 1
    span = target.shape[dim]
    if assignment.block_elems is None:
        return value.narrow(dim, rank * span, span)
    # FP8 block scales are stored one value per ``FP8_BLOCK_SIZE`` element block.
    # The per-rank element span must be block-aligned, otherwise a single scale
    # block straddles two ranks and cannot be sharded without corrupting the
    # neighbouring rank; assert so the misalignment fails loudly instead of
    # silently slicing at ``rank * ceil(span/block)`` (the prior behaviour).
    block_elems = assignment.block_elems
    if block_elems % FP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"FP8 sharded dim {block_elems} for {assignment.source_key} is not a "
            f"multiple of block size {FP8_BLOCK_SIZE}; scale blocks would misalign"
        )
    block_start = (rank * block_elems) // FP8_BLOCK_SIZE
    return value.narrow(dim, block_start, span)


def _add_linear(
    assignments: list[Assignment],
    module,
    hf_weight_key: str,
    slice_kind: SLICE,
) -> None:
    assignments.append(Assignment(hf_weight_key, module.weight, slice_kind))
    scale = getattr(module, "scale", None)
    if scale is not None:
        # The scale tensor is block-wise (one value per FP8_BLOCK_SIZE block),
        # so its rank slice cannot reuse the weight's element-unit slice math.
        # Record the weight's per-rank element span along the sharded axis so
        # ``_slice_for_rank`` can derive the block-aligned scale offset and
        # reject spans that are not a multiple of the block size.
        block_elems = None
        if slice_kind == "row":
            block_elems = module.weight.shape[0]
        elif slice_kind == "col":
            block_elems = module.weight.shape[1]
        assignments.append(
            Assignment(hf_weight_key + "_scale_inv", scale, slice_kind, block_elems)
        )


def _build_assignments(
    model: torch.nn.Module,
    *,
    rank: int,
    world_size: int,
) -> list[Assignment]:
    del rank, world_size
    assignments: list[Assignment] = []

    assignments.append(Assignment("model.embed_tokens.weight", model.embed.weight, "row"))
    assignments.append(Assignment("model.norm.weight", model.norm.weight, "full"))
    _add_linear(assignments, model.head, "lm_head.weight", "row")

    for layer_idx, layer in enumerate(model.layers):
        prefix = f"model.layers.{layer_idx}"
        assignments.append(
            Assignment(f"{prefix}.input_layernorm.weight", layer.attn_norm.weight, "full")
        )
        assignments.append(
            Assignment(f"{prefix}.post_attention_layernorm.weight", layer.ffn_norm.weight, "full")
        )

        attn = layer.attn
        attn_prefix = f"{prefix}.self_attn"
        _add_linear(assignments, attn.wq_a, f"{attn_prefix}.q_a_proj.weight", "full")
        assignments.append(
            Assignment(f"{attn_prefix}.q_a_layernorm.weight", attn.q_norm.weight, "full")
        )
        _add_linear(assignments, attn.wq_b, f"{attn_prefix}.q_b_proj.weight", "row")
        _add_linear(
            assignments,
            attn.wkv_a,
            f"{attn_prefix}.kv_a_proj_with_mqa.weight",
            "full",
        )
        assignments.append(
            Assignment(f"{attn_prefix}.kv_a_layernorm.weight", attn.kv_norm.weight, "full")
        )
        _add_linear(assignments, attn.wkv_b, f"{attn_prefix}.kv_b_proj.weight", "row")
        _add_linear(assignments, attn.wo, f"{attn_prefix}.o_proj.weight", "col")

        indexer = attn.indexer
        idx_prefix = f"{attn_prefix}.indexer"
        _add_linear(assignments, indexer.wq_b, f"{idx_prefix}.wq_b.weight", "full")
        _add_linear(assignments, indexer.wk, f"{idx_prefix}.wk.weight", "full")
        assignments.append(Assignment(f"{idx_prefix}.k_norm.weight", indexer.k_norm.weight, "full"))
        assignments.append(Assignment(f"{idx_prefix}.k_norm.bias", indexer.k_norm.bias, "full"))
        _add_linear(
            assignments,
            indexer.weights_proj,
            f"{idx_prefix}.weights_proj.weight",
            "full",
        )

        ffn_prefix = f"{prefix}.mlp"
        ffn = layer.ffn
        if hasattr(ffn, "gate"):
            assignments.append(Assignment(f"{ffn_prefix}.gate.weight", ffn.gate.weight, "full"))
            if getattr(ffn.gate, "bias", None) is not None:
                assignments.append(
                    Assignment(
                        f"{ffn_prefix}.gate.e_score_correction_bias",
                        ffn.gate.bias,
                        "full",
                    )
                )
            for expert_idx in range(ffn.experts_start_idx, ffn.experts_end_idx):
                expert = ffn.experts[expert_idx]
                expert_prefix = f"{ffn_prefix}.experts.{expert_idx}"
                _add_linear(assignments, expert.w1, f"{expert_prefix}.gate_proj.weight", "full")
                _add_linear(assignments, expert.w2, f"{expert_prefix}.down_proj.weight", "full")
                _add_linear(assignments, expert.w3, f"{expert_prefix}.up_proj.weight", "full")

            shared = ffn.shared_experts
            shared_prefix = f"{ffn_prefix}.shared_experts"
            _add_linear(assignments, shared.w1, f"{shared_prefix}.gate_proj.weight", "row")
            _add_linear(assignments, shared.w2, f"{shared_prefix}.down_proj.weight", "col")
            _add_linear(assignments, shared.w3, f"{shared_prefix}.up_proj.weight", "row")
        else:
            _add_linear(assignments, ffn.w1, f"{ffn_prefix}.gate_proj.weight", "row")
            _add_linear(assignments, ffn.w2, f"{ffn_prefix}.down_proj.weight", "col")
            _add_linear(assignments, ffn.w3, f"{ffn_prefix}.up_proj.weight", "row")

    return assignments
