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
# Each node: (id, label)
# All nodes are drawn as uniform light-gray boxes.

NODES = [
    ("input_ids",        "input_ids"),
    ("embed",            "Embedding"),
    ("attn_norm",        "Attn RMS"),
    ("qkv_proj",         "QKV A Proj"),
    ("split_qkv",        "Split"),
    ("rms_norm_q",       "Q RMS"),
    ("rms_norm_kv",      "KV C RMS"),
    ("rope_kpe",         "KV RoPE"),
    ("q_b_proj",         "Q B Proj"),
    ("rope_q",           "Q RoPE"),
    ("cat_q",            "Cat Q"),
    ("kv_b_proj",        "KV B Proj"),
    ("split_kv",         "Split KV"),
    ("cat_k",            "Cat K"),
    ("causal_attn",      "Causal Attention"),
    ("o_proj",           "O Proj"),
    ("res_add_attn",     "Add + Residual"),
    ("ffn_norm",         "FFN RMS"),
    ("gate_router",      "Gate Router"),
    ("topk_softmax",     "TopK + Softmax"),
    ("w1_proj",          "W1 (gate)"),
    ("w3_proj",          "W3 (up)"),
    ("swiglu",           "SwiGLU"),
    ("w2_proj",          "W2 (down)"),
    ("expert_sum",       "Expert Sum"),
    ("res_add_ffn",      "Add + Residual"),
    ("final_norm",       "Final RMS"),
    ("lm_head",          "LM Head"),
    ("logits",           "logits"),
]

EDGES = [
    ("input_ids",    "embed"),
    ("embed",        "attn_norm"),
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
    ("split_kv",    "causal_attn"),
    ("causal_attn", "o_proj"),
    ("o_proj",      "res_add_attn"),
    ("embed",       "res_add_attn"),
    ("res_add_attn","ffn_norm"),
    ("ffn_norm",    "gate_router"),
    ("gate_router", "topk_softmax"),
    ("topk_softmax","w1_proj"),
    ("topk_softmax","w3_proj"),
    ("ffn_norm",    "w1_proj"),
    ("ffn_norm",    "w3_proj"),
    ("w1_proj",     "swiglu"),
    ("w3_proj",     "swiglu"),
    ("swiglu",      "w2_proj"),
    ("w2_proj",     "expert_sum"),
    ("expert_sum",  "res_add_ffn"),
    ("res_add_attn","res_add_ffn"),
    ("res_add_ffn", "final_norm"),
    ("final_norm",  "lm_head"),
    ("lm_head",     "logits"),
]

# ── Fused-op cluster styling ──────────────────────────────────────────────
#
# The mapping from fused op → graph node IDs lives in each partition's
# model.py as FUSED_OP_GRAPH.  The script reads that dict and draws a
# colored cluster box around each group.

CLUSTER_COLORS = [
    "#CC0000", "#1976D2", "#2E7D32", "#E65100",
    "#7B1FA2", "#00838F", "#C62828", "#283593",
]

NODE_IDS = {nid for nid, _ in NODES}


def _read_fused_op_graph(model_py: Path) -> dict[str, list[str]]:
    """Read FUSED_OP_GRAPH from a partition model.py.

    If the partition delegates to partition_common (which uses the 3336cdbd
    model), follow that chain to read from the base partition.
    """
    if not model_py.exists():
        return {}
    text = model_py.read_text()

    graph = _parse_fused_op_graph(text)
    if graph:
        return graph

    if "partition_common" in text:
        base_model = PARTITIONS_DIR / "3336cdbd" / "model.py"
        if base_model.exists():
            return _parse_fused_op_graph(base_model.read_text())

    return {}


def _parse_fused_op_graph(text: str) -> dict[str, list[str]]:
    """Extract FUSED_OP_GRAPH dict from Python source text."""
    m = re.search(
        r"FUSED_OP_GRAPH\s*=\s*(\{.*?\})\s*\n",
        text,
        re.DOTALL,
    )
    if not m:
        return {}
    try:
        result = eval(m.group(1))  # noqa: S307
    except Exception:
        return {}
    if not isinstance(result, dict):
        return {}
    for op_name, nodes in result.items():
        for nid in nodes:
            if nid not in NODE_IDS:
                print(f"  WARNING: FUSED_OP_GRAPH[{op_name!r}] references "
                      f"unknown node {nid!r}")
    return result


def _build_graph(
    title: str,
    fused_op_graph: dict[str, list[str]] | None = None,
    partition_notes: str = "",
) -> graphviz.Digraph:
    full_label = title
    if partition_notes:
        full_label = f"{title}\n{partition_notes}"

    g = graphviz.Digraph(
        name="ops",
        format="png",
        graph_attr={
            "rankdir": "TB",
            "bgcolor": "white",
            "fontname": "Helvetica",
            "fontsize": "16",
            "label": full_label,
            "labelloc": "t",
            "labeljust": "c",
            "pad": "0.4",
            "nodesep": "0.5",
            "ranksep": "0.6",
            "dpi": "150",
            "splines": "true",
        },
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "12",
            "style": "filled",
            "shape": "box",
            "fillcolor": "#F0F0F0",
            "color": "#888888",
            "penwidth": "1.0",
            "margin": "0.12,0.06",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "9",
            "color": "#555555",
            "arrowsize": "0.8",
        },
    )

    node_lookup = {nid: nlabel for nid, nlabel in NODES}

    fused_node_set: set[str] = set()
    if fused_op_graph:
        for nodes in fused_op_graph.values():
            fused_node_set.update(nodes)

        for i, (op_name, nodes) in enumerate(fused_op_graph.items()):
            if not nodes:
                continue
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            kernel_num = i + 1
            cluster_label = f"Fused Kernel {kernel_num}: {op_name}"

            with g.subgraph(name=f"cluster_{op_name}") as sub:
                sub.attr(
                    label=cluster_label,
                    style="bold,rounded",
                    color=color,
                    fontcolor=color,
                    fontsize="16",
                    fontname="Helvetica Bold",
                    penwidth="3.0",
                    margin="20",
                    labeljust="l",
                )
                for node_id in nodes:
                    if node_id not in node_lookup:
                        continue
                    sub.node(node_id, label=node_lookup[node_id])

    for node_id, label in NODES:
        if node_id in fused_node_set:
            continue
        g.node(node_id, label=label)

    for src, dst in EDGES:
        g.edge(src, dst)

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

    fused_op_graph = _read_fused_op_graph(model_py)
    if not fused_op_graph:
        print(f"  {partition_dir.name} → skipped (no FUSED_OP_GRAPH)")
        return None

    notes = _read_partition_notes(model_py)
    title = f"DeepSeek V3.2 — Partition {partition_dir.name}"
    g = _build_graph(title, fused_op_graph=fused_op_graph, partition_notes=notes)

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
