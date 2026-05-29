"""Racetrack compile backend: torch.compile custom backend driven by PartitionSpec.

Usage:
    from racetrack.compile_backend import compile_with_partition
    from racetrack.partition_spec import load_spec

    spec = load_spec("dsv3_2", "3336cdbd")
    compiled, originals = compile_with_partition(model, spec)
    output = compiled(tokens, pos)

Or register as a named backend:
    torch.compile(model, backend="racetrack")

The backend receives FX GraphModules from Dynamo, applies FX pattern
matching to replace fusible subgraphs with custom kernel calls, and
returns gm.forward for direct execution (no Inductor). Pre-trace patches
handle module-level changes before Dynamo traces.
"""
from __future__ import annotations

import os
from typing import Callable

import torch
import torch.nn as nn
from torch.fx import GraphModule

from racetrack.fx_patterns import rewrite_graph
from racetrack.partition_spec import BASELINE_SPEC, PartitionSpec
from racetrack.pre_trace import Originals, apply_pre_trace_patches
from racetrack.runtime.dispatch import KernelDispatcher


class RacetrackBackend:
    """torch.compile custom backend driven by a PartitionSpec.

    Applies FX pattern matching to replace fusible subgraphs with
    custom kernel calls, then returns gm.forward for direct execution.
    """

    def __init__(self, spec: PartitionSpec):
        self.spec = spec
        if spec.kernel_root.is_dir():
            self.dispatcher = KernelDispatcher(spec.kernel_root)
        else:
            self.dispatcher = KernelDispatcher(None)

    def __call__(self, gm: GraphModule, example_inputs):
        gm = rewrite_graph(gm, self.spec, self.dispatcher)
        gm.recompile()
        return gm.forward


def make_racetrack_backend(
    spec: PartitionSpec | None = None,
) -> Callable:
    if spec is None:
        spec = BASELINE_SPEC
    return RacetrackBackend(spec)


def compile_with_partition(
    model: nn.Module,
    spec: PartitionSpec,
    *,
    backend_override: str | None = None,
) -> tuple[nn.Module, Originals]:
    """Apply a partition to a model via torch.compile.

    1. Creates KernelDispatcher for the partition's kernels
    2. Applies pre-trace patches (module-level changes)
    3. Compiles with RacetrackBackend (FX pattern matching)

    Returns (compiled_model, originals_for_rollback).
    """
    if backend_override is not None:
        # torch.compile is lazy: RacetrackBackend.__call__ -> rewrite_graph and
        # all runtime KernelDispatcher.call dispatch read RACETRACK_KERNEL_BACKEND
        # only when the compiled model is FIRST EXECUTED, not here. Restoring the
        # var now would mean those lazy reads see the previous value and silently
        # drop the override, so we set it and leave it set for the lifetime of the
        # returned compiled model. Callers that need the prior value restored must
        # save/restore it around their own use of the compiled model.
        os.environ["RACETRACK_KERNEL_BACKEND"] = backend_override

    if spec.kernel_root.is_dir():
        dispatcher = KernelDispatcher(spec.kernel_root)
    else:
        dispatcher = KernelDispatcher(None)

    originals = apply_pre_trace_patches(model, spec, dispatcher)
    backend = RacetrackBackend(spec)
    compiled = torch.compile(model, backend=backend)

    return compiled, originals


def cleanup_compile_state() -> None:
    dynamo = getattr(torch, "_dynamo", None)
    reset = getattr(dynamo, "reset", None)
    if callable(reset):
        reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


try:
    from torch._dynamo import register_backend
    register_backend(name="racetrack", compiler_fn=make_racetrack_backend())
except Exception:
    pass
