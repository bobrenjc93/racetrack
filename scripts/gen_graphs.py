#!/usr/bin/env python3
"""
Generate op-flow graphs for baseline models and their partitions.

Outputs:
  partitions/<model>/graph.png           -- baseline op flow
  partitions/<model>/<hash>/graph.png    -- partition op flow with fused-kernel boxes

Usage:
  python scripts/gen_graphs.py                 # generate all graphs
  python scripts/gen_graphs.py 3336cdbd        # generate only that partition (+ baselines)
  python scripts/gen_graphs.py --baseline-only # baselines only
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import graphviz

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── dsv3_2 graph ─────────────────────────────────────────────────────────

NODES_DSV3_2 = [
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

EDGES_DSV3_2 = [
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


# ── dsv3_2_nvfp4 graph ──────────────────────────────────────────────────

NODES_NVFP4 = [
    ("input_ids",         "input_ids"),
    ("embed",             "Embedding"),
    ("ar_add_rms",        "AR + Add + RMS"),
    ("qkv_a_proj",        "QKV A Proj"),
    ("indexer_k_proj",    "Indexer K"),
    ("q_rms",             "Q RMS"),
    ("q_b_proj",          "Q B Proj"),
    ("q_rope",            "Q RoPE"),
    ("cat_q",             "Cat"),
    ("q_quant_fp8",       "Q Quantize FP8"),
    ("kv_c_rms",          "KV C RMS"),
    ("kv_rope",           "KV RoPE"),
    ("kv_quant_fp8",      "KV Quant FP8"),
    ("mla_cache",         "MLA Cache"),
    ("indexer_ln",        "LayerNorm"),
    ("indexer_rope",      "RoPE"),
    ("indexer_quant_fp8", "Quantize FP8"),
    ("indexer_cache",     "Indexer Cache"),
    ("indexer_w",         "Indexer W"),
    ("indexer_q_proj",    "Indexer Q Proj"),
    ("indexer_q_rope",    "Indexer Q RoPE"),
    ("indexer_q_fp8",     "Indexer Q FP8"),
    ("w_uk_t",            "W_UK_T"),
    ("indexer_w_scale",   "Indexer W scale"),
    ("indexer_mqa",       "Indexer MQA"),
    ("logits_topk",       "Logits Top K"),
    ("topk_page_idx",     "Top K Page Indices"),
    ("mla",               "MLA"),
    ("w_uv",              "W_UV"),
    ("o_proj",            "O Proj"),
    ("ffn_norm",          "FFN RMS"),
    ("gate_router",       "Gate Router"),
    ("topk_softmax",      "TopK + Softmax"),
    ("w1_proj",           "W1 (gate)"),
    ("w3_proj",           "W3 (up)"),
    ("swiglu",            "SwiGLU"),
    ("w2_proj",           "W2 (down)"),
    ("expert_sum",        "Expert Sum"),
    ("res_add_ffn",       "Add + Residual"),
    ("final_norm",        "Final RMS"),
    ("lm_head",           "LM Head"),
    ("logits",            "logits"),
]

EDGES_NVFP4 = [
    ("input_ids",         "embed"),
    ("embed",             "ar_add_rms"),
    ("ar_add_rms",        "qkv_a_proj"),
    ("qkv_a_proj",        "q_rms"),
    ("qkv_a_proj",        "kv_c_rms"),
    ("qkv_a_proj",        "kv_rope"),
    ("ar_add_rms",        "indexer_k_proj"),
    ("indexer_k_proj",    "indexer_ln"),
    ("indexer_ln",        "indexer_rope"),
    ("indexer_rope",      "indexer_quant_fp8"),
    ("indexer_quant_fp8", "indexer_cache"),
    ("q_rms",             "q_b_proj"),
    ("q_rms",             "indexer_q_proj"),
    ("q_b_proj",          "q_rope"),
    ("q_rope",            "cat_q"),
    ("cat_q",             "q_quant_fp8"),
    ("kv_c_rms",          "kv_quant_fp8"),
    ("kv_rope",           "kv_quant_fp8"),
    ("kv_quant_fp8",      "mla_cache"),
    ("indexer_q_proj",    "indexer_q_rope"),
    ("indexer_q_rope",    "indexer_q_fp8"),
    ("indexer_q_fp8",     "indexer_w_scale"),
    ("ar_add_rms",        "indexer_w"),
    ("indexer_w",         "indexer_w_scale"),
    ("indexer_w_scale",   "indexer_mqa"),
    ("indexer_cache",     "indexer_mqa"),
    ("indexer_mqa",       "logits_topk"),
    ("logits_topk",       "topk_page_idx"),
    ("q_rms",             "w_uk_t"),
    ("w_uk_t",            "mla"),
    ("q_quant_fp8",       "mla"),
    ("mla_cache",         "mla"),
    ("topk_page_idx",     "mla"),
    ("mla",               "w_uv"),
    ("w_uv",              "o_proj"),
    ("o_proj",            "ffn_norm"),
    ("embed",             "ffn_norm"),
    ("ffn_norm",          "gate_router"),
    ("gate_router",       "topk_softmax"),
    ("topk_softmax",      "w1_proj"),
    ("topk_softmax",      "w3_proj"),
    ("ffn_norm",          "w1_proj"),
    ("ffn_norm",          "w3_proj"),
    ("w1_proj",           "swiglu"),
    ("w3_proj",           "swiglu"),
    ("swiglu",            "w2_proj"),
    ("w2_proj",           "expert_sum"),
    ("expert_sum",        "res_add_ffn"),
    ("o_proj",            "res_add_ffn"),
    ("res_add_ffn",       "final_norm"),
    ("final_norm",        "lm_head"),
    ("lm_head",           "logits"),
]


# ── Model registry ──────────────────────────────────────────────────────

MODEL_GRAPHS: dict[str, dict] = {
    "dsv3_2": {
        "dir": PROJECT_ROOT / "partitions" / "dsv3_2",
        "nodes": NODES_DSV3_2,
        "edges": EDGES_DSV3_2,
        "title": "DeepSeek V3.2",
    },
    "dsv3_2_nvfp4": {
        "dir": PROJECT_ROOT / "partitions" / "dsv3_2_nvfp4",
        "nodes": NODES_NVFP4,
        "edges": EDGES_NVFP4,
        "title": "DeepSeek V3.2 NVFP4",
    },
}

CLUSTER_COLORS = [
    "#CC0000", "#E65100", "#2E7D32", "#1976D2",
    "#7B1FA2", "#00838F", "#C62828", "#283593",
]


def _read_fused_op_graph(model_py: Path, model_dir: Path) -> dict[str, list[str]]:
    if not model_py.exists():
        return {}
    text = model_py.read_text()
    graph = _parse_fused_op_graph(text)
    if graph:
        return graph
    if "partition_common" in text:
        base_model = model_dir / "3336cdbd" / "model.py"
        if base_model.exists():
            return _parse_fused_op_graph(base_model.read_text())
    return {}


def _parse_fused_op_graph(text: str) -> dict[str, list[str]]:
    m = re.search(r"FUSED_OP_GRAPH\s*=\s*(\{.*?\})\s*\n", text, re.DOTALL)
    if not m:
        return {}
    try:
        result = eval(m.group(1))  # noqa: S307
    except Exception:
        return {}
    if not isinstance(result, dict):
        return {}
    return result


def _build_graph(
    title: str,
    nodes: list[tuple[str, str]],
    edges: list[tuple[str, str]],
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
            "nodesep": "0.35",
            "ranksep": "0.5",
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

    node_lookup = {nid: nlabel for nid, nlabel in nodes}

    fused_node_set: set[str] = set()
    if fused_op_graph:
        for fnodes in fused_op_graph.values():
            fused_node_set.update(fnodes)

        for i, (op_name, fnodes) in enumerate(fused_op_graph.items()):
            if not fnodes:
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
                for node_id in fnodes:
                    if node_id not in node_lookup:
                        continue
                    sub.node(node_id, label=node_lookup[node_id])

    for node_id, label in nodes:
        if node_id in fused_node_set:
            continue
        g.node(node_id, label=label)

    for src, dst in edges:
        g.edge(src, dst)

    return g


def _read_partition_notes(model_py: Path) -> str:
    if not model_py.exists():
        return ""
    text = model_py.read_text()
    m = re.search(
        r'PARTITION_NOTES\s*=\s*\(\s*((?:"[^"]*"\s*)+)\)',
        text, re.DOTALL,
    )
    if not m:
        m = re.search(r'PARTITION_NOTES\s*=\s*"([^"]*)"', text)
    if not m:
        return ""
    raw = m.group(1)
    parts = re.findall(r'"([^"]*)"', raw)
    return "".join(parts)


def generate_baseline_graph(model_name: str, info: dict) -> Path:
    g = _build_graph(
        f"{info['title']} — Baseline Op Flow",
        info["nodes"], info["edges"],
    )
    out = info["dir"] / "graph"
    g.render(str(out), cleanup=True)
    dest = info["dir"] / "graph.png"
    print(f"  {model_name} baseline → {dest}")
    return dest


def generate_partition_graph(
    partition_dir: Path, info: dict,
) -> Path | None:
    model_py = partition_dir / "model.py"
    if not model_py.exists():
        return None
    fused_op_graph = _read_fused_op_graph(model_py, info["dir"])
    if not fused_op_graph:
        print(f"  {partition_dir.name} → skipped (no FUSED_OP_GRAPH)")
        return None
    notes = _read_partition_notes(model_py)
    title = f"{info['title']} — Partition {partition_dir.name}"
    g = _build_graph(
        title, info["nodes"], info["edges"],
        fused_op_graph=fused_op_graph, partition_notes=notes,
    )
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

    for model_name, info in MODEL_GRAPHS.items():
        if not info["dir"].is_dir():
            continue
        generate_baseline_graph(model_name, info)
        if baseline_only:
            continue
        for child in sorted(info["dir"].iterdir()):
            if not child.is_dir():
                continue
            if child.name.startswith("_"):
                continue
            if targets and child.name not in targets:
                continue
            if not (child / "model.py").exists():
                continue
            generate_partition_graph(child, info)

    print("Done.")


if __name__ == "__main__":
    main()
