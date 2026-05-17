"""FX graph pattern matchers: replace subgraph patterns with custom kernel calls.

Each pattern matcher identifies a fusible op sequence in an FX graph and
replaces it with a call to a partition kernel wrapped as a torch.library
custom_op. Pattern matchers handle elementwise fusions that are visible
in the FX graph after Dynamo tracing (fx_pattern kind ops).
"""
from __future__ import annotations

from typing import Any, Callable

import torch
from torch.fx import GraphModule

from racetrack.partition_spec import PartitionSpec
from racetrack.runtime.dispatch import KernelDispatcher


_PATTERN_REGISTRY: dict[str, Callable] = {}
_CUSTOM_OPS: dict[str, Any] = {}


def register_pattern(name: str):
    def decorator(fn: Callable) -> Callable:
        _PATTERN_REGISTRY[name] = fn
        return fn
    return decorator


def _get_or_create_custom_op(name: str, impl_fn, fake_fn):
    if name not in _CUSTOM_OPS:
        op = torch.library.custom_op(f"racetrack::{name}", mutates_args=())(impl_fn)
        op.register_fake(fake_fn)
        _CUSTOM_OPS[name] = op
    return _CUSTOM_OPS[name]


def rewrite_graph(
    gm: GraphModule,
    spec: PartitionSpec,
    dispatcher: KernelDispatcher,
) -> GraphModule:
    graph = gm.graph
    rewrites = 0

    for op in spec.fx_ops:
        matcher = _PATTERN_REGISTRY.get(op.name)
        if matcher is None:
            continue
        count = matcher(graph, dispatcher)
        rewrites += count

    if rewrites > 0:
        graph.lint()
        gm.recompile()

    return gm


# ---------------------------------------------------------------------------
# Pattern: fused_swiglu — silu(gate) * up
# ---------------------------------------------------------------------------


@register_pattern("fused_swiglu")
def _match_and_replace_swiglu(graph: torch.fx.Graph, dispatcher: KernelDispatcher) -> int:
    kernel_fn = _resolve(dispatcher, "fused_swiglu")
    if kernel_fn is None:
        return 0

    def _impl(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return kernel_fn(gate, up, fallback=None)

    def _fake(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(gate)

    custom_op = _get_or_create_custom_op("fused_swiglu", _impl, _fake)
    op_callable = custom_op._opoverload

    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", "")
        if name != "silu":
            continue
        for user in list(node.users):
            if user.op != "call_function" or "mul" not in str(user.target):
                continue
            gate_input = node.args[0]
            other_args = [a for a in user.args if a is not node]
            if not other_args:
                continue
            up_input = other_args[0]
            with graph.inserting_before(user):
                fused = graph.call_function(op_callable, args=(gate_input, up_input))
            user.replace_all_uses_with(fused)
            graph.erase_node(user)
            if len(node.users) == 0:
                graph.erase_node(node)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Pattern: fused_residual_norm — add(residual, x) → rmsnorm chain
# ---------------------------------------------------------------------------


@register_pattern("fused_residual_norm")
def _match_and_replace_residual_norm(graph: torch.fx.Graph, dispatcher: KernelDispatcher) -> int:
    kernel_fn = _resolve(dispatcher, "fused_residual_norm")
    if kernel_fn is None:
        return 0

    def _impl(
        residual: torch.Tensor,
        update: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return kernel_fn(residual, update, weight, eps=eps, fallback=None)

    def _fake(
        residual: torch.Tensor,
        update: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(update), torch.empty_like(update)

    _get_or_create_custom_op("fused_residual_norm", _impl, _fake)
    return 0


# ---------------------------------------------------------------------------
# Pattern: fused_rms_norm — pow(2) → mean → add(eps) → rsqrt → mul → mul
# ---------------------------------------------------------------------------


@register_pattern("fused_rms_norm")
def _match_and_replace_rms_norm(graph: torch.fx.Graph, dispatcher: KernelDispatcher) -> int:
    kernel_fn = _resolve(dispatcher, "fused_rms_norm")
    if kernel_fn is None:
        kernel_fn = _resolve(dispatcher, "fused_residual_norm")
    if kernel_fn is None:
        return 0
    return 0


# ---------------------------------------------------------------------------
# Pattern: fused_act_quant — float → abs → amax → clamp → div → clamp → fp8
# ---------------------------------------------------------------------------


@register_pattern("fused_act_quant")
def _match_and_replace_act_quant(graph: torch.fx.Graph, dispatcher: KernelDispatcher) -> int:
    kernel_fn = _resolve(dispatcher, "fused_act_quant")
    if kernel_fn is None:
        return 0

    def _impl(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return kernel_fn(x, fallback=None)

    def _fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        N = shape[-1]
        n_groups = (N + 127) // 128
        return (
            torch.empty(shape, dtype=torch.float8_e4m3fn, device=x.device),
            torch.empty(*shape[:-1], n_groups, dtype=torch.float32, device=x.device),
        )

    _get_or_create_custom_op("fused_act_quant", _impl, _fake)
    return 0


# ---------------------------------------------------------------------------
# Pattern: fused_norm_rope — rmsnorm + rope
# ---------------------------------------------------------------------------


@register_pattern("fused_norm_rope")
def _match_and_replace_norm_rope(graph: torch.fx.Graph, dispatcher: KernelDispatcher) -> int:
    kernel_fn = _resolve(dispatcher, "fused_norm_rope")
    if kernel_fn is None:
        return 0
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(dispatcher: KernelDispatcher, op_name: str):
    backend = dispatcher.selected_backend(default="torch")
    if backend == "best":
        backend = dispatcher._best_fast_path.get(op_name, "triton")
    if backend in ("torch", "all"):
        return None
    return dispatcher._resolve(backend, op_name)
