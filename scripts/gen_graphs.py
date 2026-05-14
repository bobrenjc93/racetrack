#!/usr/bin/env python3
"""
Generate op-flow graphs for the baseline model and each partition.

Outputs:
  partitions/dsv3_2/graph.png           -- baseline op flow
  partitions/dsv3_2/<hash>/graph.png    -- partition op flow with fused-kernel boxes

Usage:
  python scripts/gen_graphs.py                 # generate all graphs
  python scripts/gen_graphs.py 3336cdbd        # generate only that partition (+ baseline)
  python scripts/gen_graphs.py --baseline-only # baseline only
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import graphviz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARTITIONS_DIR = PROJECT_ROOT / "partitions" / "dsv3_2"


# ── Graph definition ─────────────────────────────────────────────────────
#
# Each node: (id, label, category)
# Categories control color: "io", "linear", "norm", "rope", "attn",
#                            "moe", "activation", "residual", "misc"

NODES = [
    ("input_ids",        "input_ids",                "io"),
    ("embed",            "Embedding",                "linear"),
    ("layer_start",      "× num_layers",             "misc"),
    ("attn_norm",        "RMSNorm\n(attn_norm)",     "norm"),
    ("qkv_proj",         "Linear\n(fused_qkv_a)",    "linear"),
    ("split_qkv",        "split →\nq_c, kv_c, k_pe", "misc"),
    ("rms_norm_q",       "RMSNorm\n(q_c)",           "norm"),
    ("rms_norm_kv",      "RMSNorm\n(kv_c)",          "norm"),
    ("rope_kpe",         "RoPE\n(k_pe)",             "rope"),
    ("q_b_proj",         "Linear\n(q_b_proj)",       "linear"),
    ("rope_q",           "RoPE\n(q_pe)",             "rope"),
    ("cat_q",            "cat → q",                  "misc"),
    ("kv_b_proj",        "Linear\n(kv_b_proj)",      "linear"),
    ("split_kv",         "split →\nk_nope, v",       "misc"),
    ("cat_k",            "cat → k",                  "misc"),
    ("causal_attn",      "causal_attention\n(einsum + mask\n+ softmax + einsum)", "attn"),
    ("o_proj",           "Linear\n(o_proj)",          "linear"),
    ("res_add_attn",     "residual add",             "residual"),
    ("ffn_norm",         "RMSNorm\n(ffn_norm)",      "norm"),
    ("gate_router",      "Linear\n(gate / router)",  "moe"),
    ("topk_softmax",     "topk + softmax",           "moe"),
    ("expert_start",     "× per expert",             "misc"),
    ("w1_proj",          "Linear (w1)\ngate proj",   "linear"),
    ("w3_proj",          "Linear (w3)\nup proj",     "linear"),
    ("swiglu",           "SiLU(gate) × up\n(swiglu)", "activation"),
    ("w2_proj",          "Linear (w2)\ndown proj",   "linear"),
    ("expert_end",       "expert sum",               "misc"),
    ("res_add_ffn",      "residual add",             "residual"),
    ("layer_end",        "end layer",                "misc"),
    ("final_norm",       "RMSNorm\n(final)",         "norm"),
    ("lm_head",          "Linear\n(lm_head)",        "linear"),
    ("logits",           "logits",                   "io"),
]

EDGES = [
    ("input_ids",    "embed"),
    ("embed",        "layer_start"),
    ("layer_start",  "attn_norm"),
    ("attn_norm",    "qkv_proj"),
    ("qkv_proj",     "split_qkv"),
    ("split_qkv",   "rms_norm_q"),
    ("split_qkv",   "rms_norm_kv"),
    ("split_qkv",   "rope_kpe"),
    ("rms_norm_q",  "q_b_proj"),
    ("q_b_proj",    "rope_q"),
    ("rope_q",      "cat_q"),
    ("rms_norm_kv", "kv_b_proj"),
    ("kv_b_proj",   "split_kv"),
    ("split_kv",    "cat_k"),
    ("rope_kpe",    "cat_k"),
    ("cat_q",       "causal_attn"),
    ("cat_k",       "causal_attn"),
    ("split_kv",    "causal_attn", "v"),
    ("causal_attn", "o_proj"),
    ("o_proj",      "res_add_attn"),
    ("layer_start", "res_add_attn", "residual"),
    ("res_add_attn","ffn_norm"),
    ("ffn_norm",    "gate_router"),
    ("ffn_norm",    "expert_start"),
    ("gate_router", "topk_softmax"),
    ("topk_softmax","expert_start"),
    ("expert_start","w1_proj"),
    ("expert_start","w3_proj"),
    ("w1_proj",     "swiglu"),
    ("w3_proj",     "swiglu"),
    ("swiglu",      "w2_proj"),
    ("w2_proj",     "expert_end"),
    ("expert_end",  "res_add_ffn"),
    ("res_add_attn","res_add_ffn", "residual"),
    ("res_add_ffn", "layer_end"),
    ("layer_end",   "final_norm"),
    ("final_norm",  "lm_head"),
    ("lm_head",     "logits"),
]

CATEGORY_COLORS = {
    "io":         ("#E8F5E9", "#2E7D32"),
    "linear":     ("#E3F2FD", "#1565C0"),
    "norm":       ("#FFF3E0", "#E65100"),
    "rope":       ("#F3E5F5", "#6A1B9A"),
    "attn":       ("#FCE4EC", "#AD1457"),
    "moe":        ("#E0F7FA", "#00695C"),
    "activation": ("#FFF9C4", "#F57F17"),
    "residual":   ("#EFEBE9", "#4E342E"),
    "misc":       ("#F5F5F5", "#616161"),
}

# ── Fused-op → node groups ───────────────────────────────────────────────
#
# Maps dispatcher op names to the set of graph node IDs they fuse together.
# The script draws a colored cluster box around each group.

FUSED_OP_NODES = {
    "fused_norm_rope": {
        "label": "fused_norm_rope\n(2× RMSNorm + RoPE)",
        "color": "#D32F2F",
        "nodes": ["rms_norm_q", "rms_norm_kv", "rope_kpe"],
    },
    "fused_residual_norm": {
        "label": "fused_residual_norm\n(residual add + RMSNorm)",
        "color": "#1976D2",
        "nodes": ["res_add_attn", "ffn_norm"],
    },
    "fused_swiglu": {
        "label": "fused_swiglu\n(SiLU × up)",
        "color": "#388E3C",
        "nodes": ["swiglu"],
    },
    "hc_head": {
        "label": "hc_head\n(hydra-head mixing)",
        "color": "#7B1FA2",
        "nodes": [],
    },
}

CLUSTER_COLORS = [
    "#D32F2F", "#1976D2", "#388E3C", "#F57C00",
    "#7B1FA2", "#00838F", "#C62828", "#283593",
]


def _detect_dispatched_ops(model_py: Path) -> list[str]:
    """Parse a partition model.py and return a list of dispatcher op names.

    If the partition delegates to partition_common (which uses the 3336cdbd
    model), follow that chain to find the actual dispatched ops.
    """
    if not model_py.exists():
        return []
    text = model_py.read_text()
    ops = sorted(set(re.findall(r'dispatcher\.call\(\s*["\'](\w+)["\']', text)))
    if ops:
        return ops
    if "partition_common" in text:
        base_model = PARTITIONS_DIR / "3336cdbd" / "model.py"
        if base_model.exists():
            base_text = base_model.read_text()
            return sorted(
                set(re.findall(r'dispatcher\.call\(\s*["\'](\w+)["\']', base_text))
            )
    return []


def _build_graph(
    title: str,
    fused_ops: list[str] | None = None,
    partition_notes: str = "",
) -> graphviz.Digraph:
    g = graphviz.Digraph(
        name="ops",
        format="png",
        graph_attr={
            "rankdir": "TB",
            "bgcolor": "white",
            "fontname": "Helvetica",
            "fontsize": "14",
            "label": title,
            "labelloc": "t",
            "labeljust": "c",
            "pad": "0.5",
            "nodesep": "0.4",
            "ranksep": "0.5",
            "dpi": "150",
        },
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "11",
            "style": "filled,rounded",
            "shape": "box",
            "margin": "0.15,0.08",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "9",
            "color": "#424242",
            "arrowsize": "0.7",
        },
    )

    if partition_notes:
        g.attr(
            label=f"{title}\n{partition_notes}",
        )

    fused_node_set: set[str] = set()
    active_fusions: dict[str, dict] = {}
    if fused_ops:
        for op_name in fused_ops:
            info = FUSED_OP_NODES.get(op_name)
            if info and info["nodes"]:
                active_fusions[op_name] = info
                fused_node_set.update(info["nodes"])

    cluster_membership: dict[str, str] = {}
    for op_name, info in active_fusions.items():
        for node_id in info["nodes"]:
            cluster_membership[node_id] = op_name

    for i, (op_name, info) in enumerate(active_fusions.items()):
        color = info.get("color", CLUSTER_COLORS[i % len(CLUSTER_COLORS)])
        with g.subgraph(name=f"cluster_{op_name}") as sub:
            sub.attr(
                label=info["label"],
                style="dashed,rounded,bold",
                color=color,
                fontcolor=color,
                fontsize="12",
                fontname="Helvetica Bold",
                penwidth="2.5",
                margin="16",
            )
            for node_id in info["nodes"]:
                label = None
                category = None
                for nid, nlabel, ncat in NODES:
                    if nid == node_id:
                        label = nlabel
                        category = ncat
                        break
                if label is None:
                    continue
                fill, border = CATEGORY_COLORS.get(category, ("#FFFFFF", "#000000"))
                sub.node(
                    node_id,
                    label=label,
                    fillcolor=fill,
                    color=border,
                    penwidth="1.5",
                )

    for node_id, label, category in NODES:
        if node_id in fused_node_set:
            continue
        fill, border = CATEGORY_COLORS.get(category, ("#FFFFFF", "#000000"))
        if category == "misc":
            g.node(
                node_id,
                label=label,
                shape="plaintext",
                style="",
                fillcolor="transparent",
                fontcolor="#9E9E9E",
                fontsize="10",
            )
        else:
            g.node(
                node_id,
                label=label,
                fillcolor=fill,
                color=border,
                penwidth="1.5",
            )

    for edge in EDGES:
        src, dst = edge[0], edge[1]
        edge_label = edge[2] if len(edge) > 2 else ""
        attrs = {}
        if edge_label:
            attrs["label"] = f"  {edge_label}  "
            attrs["fontcolor"] = "#9E9E9E"
            attrs["style"] = "dashed"
            attrs["color"] = "#BDBDBD"
        g.edge(src, dst, **attrs)

    return g


def _read_partition_notes(model_py: Path) -> str:
    if not model_py.exists():
        return ""
    text = model_py.read_text()
    m = re.search(
        r'PARTITION_NOTES\s*=\s*\(\s*((?:"[^"]*"\s*)+)\)',
        text,
        re.DOTALL,
    )
    if not m:
        m = re.search(r'PARTITION_NOTES\s*=\s*"([^"]*)"', text)
    if not m:
        return ""
    raw = m.group(1)
    parts = re.findall(r'"([^"]*)"', raw)
    return "".join(parts)


def generate_baseline_graph() -> Path:
    g = _build_graph("DeepSeek V3.2 — Baseline Op Flow")
    out = PARTITIONS_DIR / "graph"
    g.render(str(out), cleanup=True)
    dest = PARTITIONS_DIR / "graph.png"
    print(f"  baseline → {dest}")
    return dest


def generate_partition_graph(partition_dir: Path) -> Path | None:
    model_py = partition_dir / "model.py"
    if not model_py.exists():
        return None

    ops = _detect_dispatched_ops(model_py)
    display_ops = [op for op in ops if FUSED_OP_NODES.get(op, {}).get("nodes")]
    if not display_ops:
        print(f"  {partition_dir.name} → skipped (no graphable fused ops)")
        return None

    notes = _read_partition_notes(model_py)
    title = f"DeepSeek V3.2 — Partition {partition_dir.name}"
    g = _build_graph(title, fused_ops=display_ops, partition_notes=notes)

    out = partition_dir / "graph"
    g.render(str(out), cleanup=True)
    dest = partition_dir / "graph.png"
    print(f"  {partition_dir.name} → {dest}")
    return dest


def main() -> None:
    args = sys.argv[1:]
    baseline_only = "--baseline-only" in args
    targets = [a for a in args if not a.startswith("--")]

    print("Generating op-flow graphs...")
    generate_baseline_graph()

    if baseline_only:
        return

    for child in sorted(PARTITIONS_DIR.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        if targets and child.name not in targets:
            continue
        if not (child / "model.py").exists():
            continue
        generate_partition_graph(child)

    print("Done.")


if __name__ == "__main__":
    main()
