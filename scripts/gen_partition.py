#!/usr/bin/env python3
"""Generate partition directories from fusion specifications.

Each --fuse argument is a comma-separated group of graph nodes to fuse
into a single kernel. If the node set matches a known recipe, the codegen
applies code patches to model.py automatically.

Usage:
  python scripts/gen_partition.py \\
      --fuse rms_norm_q,rms_norm_kv,rope_kpe \\
      --fuse res_add_attn,ffn_norm

  python scripts/gen_partition.py --model dsv3_2_nvfp4 \\
      --fuse ar_add_rms,qkv_a_proj,indexer_k_proj \\
      --fuse indexer_ln,indexer_rope,indexer_quant_fp8,indexer_cache \\
      --fuse q_rms,q_b_proj,indexer_w,indexer_q_proj,indexer_q_rope,indexer_q_fp8,w_uk_t,indexer_w_scale,indexer_mqa \\
      --fuse q_rope,cat_q,q_quant_fp8

  python scripts/gen_partition.py --list-recipes
  python scripts/gen_partition.py --list-nodes
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

BACKENDS = ("triton", "helion", "cutedsl")
BACKEND_RUNTIME_MODULE = {
    "triton": "racetrack.runtime.triton_kernels",
    "helion": "racetrack.runtime.helion_kernels",
    "cutedsl": "racetrack.runtime.cutedsl_kernels",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FusionRecipe:
    name: str
    graph_nodes: list[str]
    description: str
    extra_ops: str = ""
    model_patches: list[tuple[str, str]] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    name: str
    partitions_dir: Path
    baseline_path: Path
    valid_node_ids: set[str]
    recipes: dict[str, FusionRecipe]
    runtime_ops: set[str]
    model_name_literal: str


# ---------------------------------------------------------------------------
# dsv3_2 recipes
# ---------------------------------------------------------------------------

_DSV3_2_NODE_IDS = {
    "input_ids", "embed", "attn_norm", "qkv_proj", "split_qkv",
    "rms_norm_q", "rms_norm_kv", "rope_kpe", "q_b_proj", "rope_q",
    "cat_q", "kv_b_proj", "split_kv", "cat_k", "causal_attn",
    "o_proj", "res_add_attn", "ffn_norm", "gate_router", "topk_softmax",
    "w1_proj", "w3_proj", "swiglu", "w2_proj", "expert_sum",
    "res_add_ffn", "final_norm", "lm_head", "logits",
}

_DSV3_2_RUNTIME_OPS = {"fused_norm_rope", "fused_residual_norm", "fused_swiglu"}

_DSV3_2_RECIPES: dict[str, FusionRecipe] = {}


def _reg_dsv3_2(r: FusionRecipe) -> None:
    _DSV3_2_RECIPES[r.name] = r


_reg_dsv3_2(FusionRecipe(
    name="fused_norm_rope",
    graph_nodes=["rms_norm_q", "rms_norm_kv", "rope_kpe"],
    description="Fused Q/KV RMSNorm + K_pe RoPE (already in baseline)",
))

_reg_dsv3_2(FusionRecipe(
    name="fused_residual_norm",
    graph_nodes=["res_add_attn", "ffn_norm"],
    description="Fused residual add + FFN RMSNorm",
    extra_ops="\n".join([
        "def fused_residual_norm(",
        "    residual: torch.Tensor,",
        "    update: torch.Tensor,",
        "    norm_weight: torch.Tensor,",
        "    *,",
        "    eps: float,",
        ") -> tuple[torch.Tensor, torch.Tensor]:",
        "    hidden = residual + update",
        "    normed = rms_norm(hidden, norm_weight, eps)",
        "    return hidden, normed",
    ]),
    model_patches=[
        (
            "\n".join([
                "            residual = hidden_states",
                "            x = self.attn_norm(hidden_states)",
                "            hidden_states = residual + self.attn(x, positions)",
                "            residual = hidden_states",
                "            x = self.ffn_norm(hidden_states)",
                "            hidden_states = residual + self.ffn(x, input_ids)",
                "            return hidden_states",
            ]),
            "\n".join([
                "            residual = hidden_states",
                "            x = self.attn_norm(hidden_states)",
                "            attn_out = self.attn(x, positions)",
                "            if self.dispatcher is None:",
                "                hidden_states, x = fused_residual_norm(",
                "                    residual, attn_out, self.ffn_norm.weight,",
                "                    eps=self.config.rms_norm_eps,",
                "                )",
                "            else:",
                "                hidden_states, x = self.dispatcher.call(",
                '                    "fused_residual_norm", fused_residual_norm,',
                "                    residual, attn_out, self.ffn_norm.weight,",
                "                    eps=self.config.rms_norm_eps,",
                "                )",
                "            residual = hidden_states",
                "            hidden_states = residual + self.ffn(x, input_ids)",
                "            return hidden_states",
            ]),
        ),
    ],
))

_reg_dsv3_2(FusionRecipe(
    name="fused_swiglu",
    graph_nodes=["swiglu"],
    description="Fused SwiGLU activation (SiLU gate * up, no clamp)",
    extra_ops="\n".join([
        "def fused_swiglu(",
        "    gate: torch.Tensor,",
        "    up: torch.Tensor,",
        ") -> torch.Tensor:",
        "    return F.silu(gate) * up",
    ]),
    model_patches=[
        (
            "        return self.w2(swiglu(self.w1(x), self.w3(x)))",
            "        return self.w2(fused_swiglu(self.w1(x), self.w3(x)))",
        ),
    ],
))


# ---------------------------------------------------------------------------
# dsv3_2_nvfp4 recipes (loaded from recipes_nvfp4.py)
# ---------------------------------------------------------------------------

from recipes_nvfp4 import (  # noqa: E402
    NVFP4_BASELINE,
    NVFP4_NODE_IDS,
    NVFP4_PARTITIONS_DIR,
    NVFP4_RECIPES,
    NVFP4_RUNTIME_OPS,
)

_NVFP4_RECIPES: dict[str, FusionRecipe] = {}
for _name, _data in NVFP4_RECIPES.items():
    _NVFP4_RECIPES[_name] = FusionRecipe(
        name=_name,
        graph_nodes=_data["graph_nodes"],
        description=_data["description"],
        extra_ops=_data.get("extra_ops", ""),
        model_patches=_data.get("model_patches", []),
        requires=_data.get("requires", []),
    )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, ModelConfig] = {
    "dsv3_2": ModelConfig(
        name="dsv3_2",
        partitions_dir=PROJECT_ROOT / "partitions" / "dsv3_2",
        baseline_path=PROJECT_ROOT / "partitions" / "dsv3_2" / "model.py",
        valid_node_ids=_DSV3_2_NODE_IDS,
        recipes=_DSV3_2_RECIPES,
        runtime_ops=_DSV3_2_RUNTIME_OPS,
        model_name_literal='MODEL_NAME = "dsv3_2"',
    ),
    "dsv3_2_nvfp4": ModelConfig(
        name="dsv3_2_nvfp4",
        partitions_dir=NVFP4_PARTITIONS_DIR,
        baseline_path=NVFP4_BASELINE,
        valid_node_ids=NVFP4_NODE_IDS,
        recipes=_NVFP4_RECIPES,
        runtime_ops=NVFP4_RUNTIME_OPS,
        model_name_literal='MODEL_NAME = "dsv3_2_nvfp4"',
    ),
}

DEFAULT_MODEL = "dsv3_2"


# ---------------------------------------------------------------------------
# Fusion spec parsing
# ---------------------------------------------------------------------------


@dataclass
class FusionSpec:
    name: str
    graph_nodes: list[str]
    recipe: FusionRecipe | None


def parse_fuse_arg(arg: str, model: ModelConfig) -> FusionSpec:
    if ":" in arg:
        nodes_str, name = arg.rsplit(":", 1)
        nodes = [n.strip() for n in nodes_str.split(",")]
    elif arg in model.recipes:
        r = model.recipes[arg]
        return FusionSpec(name=r.name, graph_nodes=list(r.graph_nodes), recipe=r)
    else:
        nodes = [n.strip() for n in arg.split(",")]
        name = "fused_" + "_".join(nodes)

    for nid in nodes:
        if nid not in model.valid_node_ids:
            print(f"ERROR: Unknown graph node '{nid}' for model {model.name}",
                  file=sys.stderr)
            print(f"Valid nodes: {', '.join(sorted(model.valid_node_ids))}",
                  file=sys.stderr)
            sys.exit(1)

    recipe = model.recipes.get(name)
    if recipe is None:
        target = set(nodes)
        for r in model.recipes.values():
            if set(r.graph_nodes) == target:
                recipe = r
                name = r.name
                break

    return FusionSpec(name=name, graph_nodes=nodes, recipe=recipe)


def validate_requirements(fusions: list[FusionSpec], model: ModelConfig) -> None:
    specified = {f.name for f in fusions}
    for f in fusions:
        if f.recipe is None:
            continue
        for req in f.recipe.requires:
            if req not in specified:
                print(
                    f"ERROR: Recipe '{f.name}' requires '{req}' "
                    f"(these fusions are interdependent).\n"
                    f"Add --fuse {req} to the command.",
                    file=sys.stderr,
                )
                sys.exit(1)


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


KERNEL_DISPATCH_MARKER = (
    "# ---------------------------------------------------------------------------\n"
    "# Kernel dispatch\n"
    "# ---------------------------------------------------------------------------"
)


def _format_fused_op_graph(graph: dict[str, list[str]]) -> str:
    if not graph:
        return "FUSED_OP_GRAPH = {}"
    lines = ["FUSED_OP_GRAPH = {"]
    for name, nodes in graph.items():
        nodes_str = ", ".join(f'"{n}"' for n in nodes)
        lines.append(f'    "{name}": [{nodes_str}],')
    lines.append("}")
    return "\n".join(lines)


def generate_model_py(
    baseline: str,
    fusions: list[FusionSpec],
    model: ModelConfig,
    cli_args: list[str],
    notes: str | None = None,
) -> str:
    text = baseline

    fused_op_graph: dict[str, list[str]] = {}
    for f in fusions:
        fused_op_graph[f.name] = f.graph_nodes

    if notes is None:
        recipe_names = [f.name for f in fusions]
        notes = f"Codegen partition: {', '.join(recipe_names)}."

    cmd = "python scripts/gen_partition.py " + " ".join(cli_args)
    text = f"# Generated by: {cmd}\n" + text

    graph_str = _format_fused_op_graph(fused_op_graph)
    old_model_name = model.model_name_literal + "\n"
    new_model_name = (
        f"{model.model_name_literal}\n"
        f"PARTITION_NOTES = {notes!r}\n"
        f"{graph_str}\n"
    )
    text = text.replace(old_model_name, new_model_name, 1)

    extra_ops_parts = [
        f.recipe.extra_ops
        for f in fusions
        if f.recipe and f.recipe.extra_ops
    ]
    if extra_ops_parts:
        if KERNEL_DISPATCH_MARKER not in text:
            print(
                "ERROR: Could not find Kernel dispatch section marker in baseline.",
                file=sys.stderr,
            )
            sys.exit(1)
        joined = "\n\n\n".join(extra_ops_parts)
        text = text.replace(
            KERNEL_DISPATCH_MARKER,
            joined + "\n\n\n" + KERNEL_DISPATCH_MARKER,
            1,
        )

    for f in fusions:
        if f.recipe is None:
            continue
        for old, new in f.recipe.model_patches:
            if old not in text:
                print(
                    f"ERROR: Patch for '{f.name}' did not match baseline text.\n"
                    f"The baseline model.py may have changed. Update the recipe.",
                    file=sys.stderr,
                )
                sys.exit(1)
            text = text.replace(old, new, 1)

    old_build = "    return FlattenedDeepSeekModel(config)\n"
    new_build = (
        "    return FlattenedDeepSeekModel("
        "config, partition_root=Path(__file__).parent)\n"
    )
    if old_build in text:
        text = text.replace(old_build, new_build, 1)

    return text


_FX_PATTERN_OPS = {
    "fused_swiglu", "fused_residual_norm", "fused_act_quant",
    "fused_rms_norm", "fused_norm_rope",
}
_PRE_TRACE_OPS = {
    "fused_full_topk_indexer", "fused_single_token_moe",
}


def _classify_op_kind(op_name: str) -> str:
    if op_name in _FX_PATTERN_OPS:
        return "fx_pattern"
    if op_name in _PRE_TRACE_OPS:
        return "pre_trace"
    return "module_patch"


def generate_spec_py(
    fusions: list[FusionSpec],
    model: ModelConfig,
    notes: str | None = None,
) -> str:
    if notes is None:
        recipe_names = [f.name for f in fusions]
        notes = f"Codegen partition: {', '.join(recipe_names)}."

    graph_nodes: dict[str, list[str]] = {}
    for f in fusions:
        graph_nodes[f.name] = f.graph_nodes

    graph_lines = ["GRAPH_NODES = {"]
    for name, nodes in graph_nodes.items():
        nodes_str = ", ".join(f'"{n}"' for n in nodes)
        graph_lines.append(f'    "{name}": [{nodes_str}],')
    graph_lines.append("}")

    ops_lines = ["FUSED_OPS = ["]
    for f in fusions:
        kind = _classify_op_kind(f.name)
        ops_lines.append(f'    {{"name": "{f.name}", "kind": "{kind}"}},')
    ops_lines.append("]")

    return "\n".join([
        f"PARTITION_NOTES = {notes!r}",
        "",
        *graph_lines,
        "",
        *ops_lines,
        "",
    ])


def generate_kernel_stub(
    backend: str, op_names: list[str], runtime_ops: set[str],
) -> str:
    runtime_module = BACKEND_RUNTIME_MODULE[backend]
    available = [op for op in op_names if op in runtime_ops]
    unavailable = [op for op in op_names if op not in runtime_ops]

    lines: list[str] = []

    if available:
        lines.append(f"from {runtime_module} import (")
        lines.append("    BACKEND_AVAILABLE,")
        for op in available:
            lines.append(f"    {op},")
        lines.append(")")
        lines.append("")
        lines.append("__all__ = [")
        lines.append('    "BACKEND_AVAILABLE",')
        for op in available:
            lines.append(f'    "{op}",')
        lines.append("]")
    else:
        lines.append("BACKEND_AVAILABLE = False")

    if unavailable:
        lines.append("")
        for op in unavailable:
            lines.append(f"# TODO: implement {backend} kernel for {op}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a partition directory from fusion specifications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/gen_partition.py \\\n"
            "      --fuse rms_norm_q,rms_norm_kv,rope_kpe \\\n"
            "      --fuse res_add_attn,ffn_norm\n\n"
            "  python scripts/gen_partition.py --model dsv3_2_nvfp4 \\\n"
            "      --fuse ar_add_rms,qkv_a_proj,indexer_k_proj \\\n"
            "      --fuse indexer_ln,indexer_rope,indexer_quant_fp8,indexer_cache \\\n"
            "      --fuse q_rope,cat_q,q_quant_fp8\n\n"
            "  python scripts/gen_partition.py --list-nodes\n"
        ),
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS.keys()),
        default=DEFAULT_MODEL,
        help=f"Model to generate partition for (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--fuse",
        action="append",
        dest="fuse",
        metavar="NODES",
        help="Comma-separated group of graph nodes to fuse (repeatable)",
    )
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument(
        "--list-nodes",
        action="store_true",
        help="List available graph nodes for the model and exit",
    )
    parser.add_argument(
        "--list-recipes",
        action="store_true",
        help="List known fusion recipes and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing files",
    )
    args = parser.parse_args()

    model = MODEL_CONFIGS[args.model]

    if args.list_nodes:
        print(f"Graph nodes for {model.name}:\n")
        for nid in sorted(model.valid_node_ids):
            print(f"  {nid}")
        print(f"\n{len(model.valid_node_ids)} nodes total.")
        return

    if args.list_recipes:
        print(f"Available fusion recipes for {model.name}:\n")
        for recipe in model.recipes.values():
            nodes = ", ".join(recipe.graph_nodes)
            has_patches = bool(recipe.model_patches or recipe.extra_ops)
            requires = ", ".join(recipe.requires) if recipe.requires else ""
            print(f"  {recipe.name}")
            print(f"    Nodes:       {nodes}")
            print(f"    Has patches: {'yes' if has_patches else 'no (baseline already wired)'}")
            if requires:
                print(f"    Requires:    {requires}")
            print(f"    {recipe.description}")
            print()
        return

    if not args.fuse:
        parser.error("at least one --fuse argument is required (or use --list-recipes)")

    cli_args: list[str] = ["--model", args.model]
    for spec in args.fuse:
        cli_args.extend(["--fuse", spec])
    if args.notes:
        cli_args.extend(["--notes", f'"{args.notes}"'])

    fusions = [parse_fuse_arg(spec, model) for spec in args.fuse]
    validate_requirements(fusions, model)
    op_names = [f.name for f in fusions]
    no_recipe = [f for f in fusions if f.recipe is None]

    print(f"Model: {model.name}")
    print("Fusions:")
    for f in fusions:
        status = "recipe" if f.recipe else "graph-only"
        print(f"  {f.name}: [{', '.join(f.graph_nodes)}] ({status})")

    if no_recipe:
        print(
            "\nWARNING: No recipe for these fusions — model.py won't be patched:"
        )
        for f in no_recipe:
            print(f"  {f.name}")

    spec_text = generate_spec_py(fusions, model, notes=args.notes)
    partition_hash = hashlib.sha256(spec_text.encode()).hexdigest()[:8]
    partition_dir = model.partitions_dir / partition_hash

    print(f"\nPartition: {partition_hash}")
    print(f"Directory: {partition_dir}")

    if args.dry_run:
        print("\nDry run — files that would be created:")
        print(f"  {partition_dir}/spec.py")
        for backend in BACKENDS:
            print(f"  {partition_dir}/kernels/{backend}/ops.py")
        print(f"  {partition_dir}/graph.png")
        return

    if partition_dir.exists():
        print("\nDirectory already exists — overwriting.")

    for backend in BACKENDS:
        (partition_dir / "kernels" / backend).mkdir(parents=True, exist_ok=True)

    (partition_dir / "spec.py").write_text(spec_text)
    print("  wrote spec.py")

    for backend in BACKENDS:
        stub = generate_kernel_stub(backend, op_names, model.runtime_ops)
        (partition_dir / "kernels" / backend / "ops.py").write_text(stub)
        print(f"  wrote kernels/{backend}/ops.py")

    gen_graphs = PROJECT_ROOT / "scripts" / "gen_graphs.py"
    try:
        subprocess.run(
            [sys.executable, str(gen_graphs), partition_hash],
            check=True,
            cwd=str(PROJECT_ROOT),
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  WARNING: graph generation failed: {e}", file=sys.stderr)

    print(f"\nDone. Test with:")
    print(
        f"  RACETRACK_KERNEL_BACKEND=triton \\\n"
        f"    python -m racetrack.bench \\\n"
        f"    --partition {partition_hash} \\\n"
        f"    --benchmark smoke"
    )


if __name__ == "__main__":
    main()
