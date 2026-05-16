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

    Strategy: DON'T replace elementwise fusions that Inductor already
    handles well (swiglu, act_quant, norm). Instead, inject ALGORITHMIC
    optimizations as custom ops — patterns where hand-written kernels
    have structural advantages over auto-generated code.

    Inductor handles the elementwise fusion + scheduling (CUDA graphs,
    dispatch overhead). We handle the algorithmic shortcuts.
    """
    graph = gm.graph
    rewrites = 0

    # --- act_quant fusion ---
    # Match: float → view → abs → amax → clamp → div(448) → div(scale) → clamp → to(fp8) → view
    # Replace with: fused_act_quant(x) → (fp8, scale)
    aq_mod = _load_kernel(partition_root, "act_quant")
    if aq_mod is not None:
        def _aq_impl(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return aq_mod.fused_act_quant(x, fallback=None)

        def _aq_fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            shape = x.shape
            N = shape[-1]
            n_groups = N // 128
            return (
                torch.empty(shape, dtype=torch.float8_e4m3fn, device=x.device),
                torch.empty(*shape[:-1], n_groups, dtype=torch.float32, device=x.device),
            )

        aq_op = _get_custom_op("fused_act_quant", _aq_impl, _aq_fake)

    # --- swiglu fusion ---
    swiglu_mod = _load_kernel(partition_root, "swiglu")
    if swiglu_mod is not None:
        def _swiglu_impl(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            return swiglu_mod.fused_swiglu(gate, up, fallback=None)

        def _swiglu_fake(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(gate)

        swiglu_op = _get_custom_op("fused_swiglu", _swiglu_impl, _swiglu_fake)
        swiglu_callable = swiglu_op._opoverload

        for silu_node, mul_node in _match_swiglu(graph):
            gate_input = silu_node.args[0]
            other_args = [a for a in mul_node.args if a is not silu_node]
            if not other_args:
                continue
            up_input = other_args[0]

            with graph.inserting_before(mul_node):
                fused = graph.call_function(swiglu_callable, args=(gate_input, up_input))
            mul_node.replace_all_uses_with(fused)
            graph.erase_node(mul_node)
            if len(silu_node.users) == 0:
                graph.erase_node(silu_node)
            rewrites += 1

    # --- residual_norm fusion ---
    # Match: add(residual, x) → pow(2) → mean → add(eps) → rsqrt → mul → mul(weight)
    norm_mod = _load_kernel(partition_root, "residual_norm")
    if norm_mod is not None:
        def _norm_impl(residual: torch.Tensor, update: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
            return norm_mod.fused_residual_norm(residual, update, weight, eps=eps, fallback=None)

        def _norm_fake(residual: torch.Tensor, update: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.empty_like(update), torch.empty_like(update)

        norm_op = _get_custom_op("fused_residual_norm", _norm_impl, _norm_fake)

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

    # Phase 1: Pattern-match and replace with our fused kernels
    gm = rewrite_graph(gm, partition_root)

    # Phase 2: Codegen a direct function from the graph.
    # No Inductor — we generate a Python function that calls
    # our kernels + PyTorch ops directly, suitable for CUDA graph capture.
    gm.recompile()

    # Return gm.forward directly — Dynamo calls this without
    # the FX interpreter overhead since gm.recompile() already
    # generated a Python function from the graph.
    return gm.forward


def patch_model_for_racetrack(model: torch.nn.Module):
    """
    Pre-trace patching: replace model functions with custom-op versions
    BEFORE Dynamo traces. This prevents decomposition of ops we want
    to keep as fused kernels.

    Call this before torch.compile(model, backend="racetrack").
    """
    partition_root = _DEFAULT_PARTITION
    rope_mod = _load_kernel(partition_root, "rope")

    if rope_mod is not None:
        import inference.model as real_model

        @torch.library.custom_op("racetrack::apply_rotary_emb", mutates_args=())
        def _rope_op(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            return rope_mod.fused_rope(x, freqs_cis, fallback=None)

        @_rope_op.register_fake
        def _(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        _original_rope = real_model.apply_rotary_emb

        def _patched_rope(x, freqs_cis, interleaved=True):
            if not interleaved:
                return _original_rope(x, freqs_cis, interleaved)
            return _rope_op(x, freqs_cis)

        real_model.apply_rotary_emb = _patched_rope


# Register as a named backend so torch.compile(backend="racetrack") works
from torch._dynamo import register_backend
register_backend(name="racetrack", compiler_fn=racetrack_backend)
