"""Graph-level partition system for racetrack.

The FX graph is the representation. A "partition" defines how the graph
is cut into subgraphs, where each subgraph becomes a compiled kernel.
The partition hash identifies the specific cutting strategy.

Usage:
    compiled = torch.compile(model, backend=make_racetrack_backend(partition))

Inductor's codegen is used for each subgraph — we're not reimplementing
codegen, we're controlling the PARTITIONING. Since Inductor generates
optimal Triton for any subgraph, our job is to find the best graph cuts.

Key insight: Inductor's default partitioning is at graph breaks (pybind
ops, dynamic control flow). We can potentially find better partitions by:
1. Merging subgraphs that Inductor keeps separate
2. Using our hand-written kernels for specific subgraph patterns
3. Fusing across boundaries Inductor doesn't see (e.g., norm → GEMM)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch.fx import GraphModule


@dataclass
class GraphPartition:
    """
    Defines how an FX graph should be partitioned into compiled subgraphs.

    The partition_id is the hash of the strategy, used to identify
    cached compilations and leaderboard entries.
    """
    name: str
    description: str
    # Map of subgraph pattern → custom kernel to use instead of Inductor codegen
    custom_kernels: dict[str, Callable] = field(default_factory=dict)
    # Whether to merge adjacent subgraphs (Inductor keeps them separate at breaks)
    merge_across_breaks: bool = False

    @property
    def partition_id(self) -> str:
        content = f"{self.name}:{self.description}:{sorted(self.custom_kernels.keys())}:{self.merge_across_breaks}"
        return hashlib.sha256(content.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Built-in partitions
# ---------------------------------------------------------------------------

BASELINE_PARTITION = GraphPartition(
    name="baseline",
    description="Passthrough to Inductor — same graph cuts, same codegen",
)


def make_racetrack_backend(partition: GraphPartition | None = None):
    """
    Create a torch.compile backend for the given partition strategy.

    Returns a backend function compatible with torch.compile(backend=...).
    """
    if partition is None:
        partition = BASELINE_PARTITION

    def backend(gm: GraphModule, example_inputs):
        # Apply partition-specific graph transformations
        gm = apply_partition(gm, partition)

        # Compile each subgraph with Inductor
        from torch._inductor.compile_fx import compile_fx
        return compile_fx(gm, example_inputs)

    return backend


def apply_partition(gm: GraphModule, partition: GraphPartition) -> GraphModule:
    """Apply partition-specific FX graph transformations."""
    if not partition.custom_kernels:
        return gm
    return gm


def apply_pre_trace_patches(model: torch.nn.Module, partition: GraphPartition):
    """
    Modify the model BEFORE Dynamo traces it.

    This controls what Dynamo sees → determines the FX graph →
    determines what Inductor compiles. Pre-trace patching can
    eliminate entire subgraphs by shortcircuiting model logic.
    """
    if "indexer_shortcircuit" in partition.custom_kernels:
        _patch_indexer_shortcircuit(model)


def _patch_indexer_shortcircuit(model: torch.nn.Module):
    """Replace Indexer.forward with shortcircuit for seq < topk.

    When end_pos <= index_topk, skip the ENTIRE indexer (wq_b, wk,
    LayerNorm, RoPE, Hadamard, FP8 quant, fp8_index, topk, broadcast
    = ~20 kernel launches) and return arange(end_pos) directly.
    Dynamo traces the simple path → Inductor compiles a tiny graph.
    """
    from inference import model as rm

    for module in model.modules():
        if not isinstance(module, rm.Indexer):
            continue
        original = module.forward
        topk = module.index_topk

        def _make_forward(orig, top_k):
            def forward(x, qr, start_pos, freqs_cis, mask):
                bsz, seqlen, _ = x.size()
                end_pos = start_pos + seqlen
                if end_pos <= top_k:
                    return torch.arange(
                        end_pos, device=x.device, dtype=torch.long,
                    ).view(1, 1, end_pos).expand(bsz, seqlen, end_pos)
                return orig(x, qr, start_pos, freqs_cis, mask)
            return forward

        module.forward = _make_forward(original, topk)


# Built-in partitions
INDEXER_SHORTCIRCUIT_PARTITION = GraphPartition(
    name="indexer_shortcircuit",
    description="Skip indexer when seq < topk — eliminates ~20 kernels/layer",
    custom_kernels={"indexer_shortcircuit": True},
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

# Register the baseline as a named backend
from torch._dynamo import register_backend

register_backend(
    name="racetrack",
    compiler_fn=make_racetrack_backend(BASELINE_PARTITION),
)
