"""Racetrack compile backend: replace FX subgraphs with partition kernels.

Usage:
    @torch.compile(backend="racetrack")
    def decode(model, tokens, pos):
        return model(tokens, pos)

Or:
    from racetrack.compile_backend import racetrack_backend
    compiled = torch.compile(model, backend=racetrack_backend)

The backend receives FX GraphModules from Dynamo, pattern-matches
for fusible op sequences, and replaces them with calls to partition
kernels (Triton/Helion/CuteDSL). The result is then executed by
Inductor or returned for eager execution.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch
from torch._dynamo.backends.common import aot_autograd
from torch.fx import GraphModule


# ---------------------------------------------------------------------------
# Kernel registry: load partition kernels by name
# ---------------------------------------------------------------------------

_KERNEL_CACHE: dict[str, Any] = {}
_CUSTOM_OPS: dict[str, Any] = {}


def _load_kernel(partition_root: Path, name: str):
    """Load a kernel module from a partition's kernel directory."""
    key = f"{partition_root}/{name}"
    if key not in _KERNEL_CACHE:
        for backend_dir in ("triton", "cutedsl", "helion"):
            path = partition_root / backend_dir / f"{name}.py"
            if path.exists():
                spec = importlib.util.spec_from_file_location(name, str(path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if getattr(mod, "BACKEND_AVAILABLE", False):
                    _KERNEL_CACHE[key] = mod
                    break
        if key not in _KERNEL_CACHE:
            _KERNEL_CACHE[key] = None
    return _KERNEL_CACHE[key]


def _get_custom_op(name: str, kernel_fn, fake_fn):
    """
    Wrap a partition kernel as a torch.library custom op so Inductor
    can trace through it with FakeTensors.
    """
    if name not in _CUSTOM_OPS:
        op = torch.library.custom_op(f"racetrack::{name}", mutates_args=())(kernel_fn)
        op.register_fake(fake_fn)
        _CUSTOM_OPS[name] = op
    return _CUSTOM_OPS[name]


# ---------------------------------------------------------------------------
# Pattern matchers: identify fusible subgraphs in the FX graph
# ---------------------------------------------------------------------------

def _match_act_quant(graph: torch.fx.Graph):
    """
    Match the act_quant pattern:
      float → reshape → abs → amax → clamp → div → div → clamp → to(fp8) → view

    Returns list of (start_node, end_node, input_node) tuples.
    """
    matches = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if "convert_element_type" in target or "_to_copy" in target:
            chain = _trace_act_quant_chain(node)
            if chain is not None:
                matches.append(chain)
    return matches


def _trace_act_quant_chain(start_node):
    """
    Trace from a float cast forward looking for the act_quant pattern.
    Returns (nodes_to_replace, input_tensor, fp8_output, scale_output) or None.
    """
    # TODO: implement full pattern matching
    return None


def _match_rmsnorm(graph: torch.fx.Graph):
    """Match: pow(2) → mean → add(eps) → rsqrt → mul(x) → mul(weight)"""
    matches = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        name = getattr(node.target, '__name__', '')
        if name == "rsqrt":
            matches.append(node)
    return matches


def _match_swiglu(graph: torch.fx.Graph):
    """Match: silu(gate) * up"""
    matches = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        name = getattr(node.target, '__name__', '')
        if name == "silu":
            for user in node.users:
                if user.op == "call_function" and "mul" in str(user.target):
                    matches.append((node, user))
    return matches


# ---------------------------------------------------------------------------
# Graph rewriter: replace matched patterns with kernel calls
# ---------------------------------------------------------------------------

def rewrite_graph(gm: GraphModule, partition_root: Path) -> GraphModule:
    """
    Rewrite an FX graph by replacing matched op patterns with
    partition kernel calls.
    """
    graph = gm.graph

    # Load available kernels
    act_quant_mod = _load_kernel(partition_root, "act_quant")
    residual_norm_mod = _load_kernel(partition_root, "residual_norm")
    swiglu_mod = _load_kernel(partition_root, "swiglu")

    rewrites = 0

    # Match and replace swiglu: silu(x) * y → fused_swiglu(x, y)
    if swiglu_mod is not None:
        def _swiglu_impl(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            return swiglu_mod.fused_swiglu(gate, up, fallback=None)

        def _swiglu_fake(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(gate)

        swiglu_op = _get_custom_op("fused_swiglu", _swiglu_impl, _swiglu_fake)

        for silu_node, mul_node in _match_swiglu(graph):
            gate_input = silu_node.args[0]
            # The mul has silu output as one arg, up as the other
            other_args = [a for a in mul_node.args if a is not silu_node]
            if not other_args:
                continue
            up_input = other_args[0]

            # Use the OpOverload (not CustomOpDef) so FX can serialize the node
            op_callable = swiglu_op._opoverload
            with graph.inserting_before(mul_node):
                fused = graph.call_function(op_callable, args=(gate_input, up_input))
            mul_node.replace_all_uses_with(fused)
            graph.erase_node(mul_node)
            if len(silu_node.users) == 0:
                graph.erase_node(silu_node)
            rewrites += 1

    if rewrites > 0:
        graph.lint()
        gm.recompile()

    return gm


# ---------------------------------------------------------------------------
# Backend entry point
# ---------------------------------------------------------------------------

# Default partition to use for kernels
_DEFAULT_PARTITION = Path("partitions/dsv3_2/3336cdbd/kernels")


def racetrack_backend(gm: GraphModule, example_inputs):
    """
    Custom torch.compile backend that replaces FX subgraphs with
    partition kernel calls, then falls through to Inductor for the rest.
    """
    partition_root = _DEFAULT_PARTITION

    # Phase 1: Pattern-match and replace with our kernels
    gm = rewrite_graph(gm, partition_root)

    # Phase 2: Let Inductor handle everything else
    # (compilation, fusion of remaining ops, CUDA graph capture)
    from torch._inductor.compile_fx import compile_fx
    return compile_fx(gm, example_inputs)


# Register as a named backend so torch.compile(backend="racetrack") works
from torch._dynamo import register_backend
register_backend(name="racetrack", compiler_fn=racetrack_backend)
