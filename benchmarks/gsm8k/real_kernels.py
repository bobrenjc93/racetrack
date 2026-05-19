"""Attach partition-local kernels to real DeepSeek inference modules.

Uses the new partition spec system: each partition is defined by a spec.py
(which ops to fuse) + kernels/<backend>/<op>.py (implementations). The
compile_with_partition() function applies pre-trace patches and wraps the
model with a torch.compile custom backend for FX pattern matching.

A leaderboard row is:
    real full model + one partition spec + one backend
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from racetrack.compile_backend import (
    cleanup_compile_state,
    compile_with_partition,
)
from racetrack.partition_spec import (
    PartitionSpec,
    discover_partitions,
    load_spec,
)
from racetrack.pre_trace import Originals, rollback_patches
from racetrack.runtime.dispatch import KernelDispatcher


from benchmarks.common import (
    TORCH_COMPILE_BACKEND,
    CONCRETE_BACKENDS,
    normalize_backend_name,
)

BACKENDS = ("torch", TORCH_COMPILE_BACKEND, *CONCRETE_BACKENDS, "best")
REAL_DISABLED_BACKENDS = {
    "dsv3_2": frozenset({"helion"}),
}


@dataclass(frozen=True)
class RealKernelRow:
    partition_model: str
    partition: str
    backend: str
    kernel_root: Path | None
    ops: tuple[str, ...]
    spec: PartitionSpec | None = None

    @property
    def label(self) -> str:
        return f"{self.partition}/{self.backend}"


@dataclass
class PatchStats:
    calls: dict[str, int]
    selected_backends: dict[str, tuple[str, ...]]

    @property
    def total_calls(self) -> int:
        return sum(self.calls.values())

    @property
    def used_partition_kernel(self) -> bool:
        return any(
            backend != "torch"
            for backends in self.selected_backends.values()
            for backend in backends
        )


def discover_real_kernel_rows(
    *,
    partition_model: str = "dsv3_2_nvfp4",
    partition_filter: str = "all",
    backend_filter: str = "all",
) -> list[RealKernelRow]:
    rows = [
        RealKernelRow(
            partition_model=partition_model,
            partition="baseline",
            backend="torch",
            kernel_root=None,
            ops=(),
        ),
    ]
    include_compile = backend_filter == "all" or any(
        _normalize_backend_name(b) == TORCH_COMPILE_BACKEND
        for b in backend_filter.split(",")
        if b.strip()
    )
    if include_compile:
        rows.append(
            RealKernelRow(
                partition_model=partition_model,
                partition="baseline",
                backend=TORCH_COMPILE_BACKEND,
                kernel_root=None,
                ops=(),
            ),
        )

    specs = discover_partitions(partition_model)
    if partition_filter != "all":
        wanted = {p.strip() for p in partition_filter.split(",") if p.strip()}
        specs = [s for s in specs if s.partition_hash in wanted]
        if not specs:
            raise KeyError(f"No partitions matched filter {partition_filter!r}")

    disabled = REAL_DISABLED_BACKENDS.get(partition_model, frozenset())

    for spec in specs:
        concrete_backends = [
            b for b in ("triton", "cutedsl", "helion")
            if b not in disabled and (spec.kernel_root / b).is_dir()
        ]
        backends = list(concrete_backends)
        if len(backends) > 1:
            backends.append("best")

        if backend_filter != "all":
            wanted_backends = {
                _normalize_backend_name(b)
                for b in backend_filter.split(",")
                if b.strip()
            }
            backends = [b for b in backends if b in wanted_backends]

        for backend in backends:
            ops = tuple(sorted(op.name for op in spec.fused_ops))
            if not ops:
                continue
            rows.append(
                RealKernelRow(
                    partition_model=partition_model,
                    partition=spec.partition_hash,
                    backend=backend,
                    kernel_root=spec.kernel_root,
                    ops=ops,
                    spec=spec,
                )
            )
    return rows


_normalize_backend_name = normalize_backend_name


@contextlib.contextmanager
def patch_real_model(
    model: torch.nn.Module,
    row: RealKernelRow,
    *,
    strict_kernel_use: bool = True,
) -> Iterator[PatchStats]:
    if row.backend == "torch" or row.kernel_root is None:
        yield PatchStats(calls={}, selected_backends={})
        return

    stats = PatchStats(
        calls={op_name: 0 for op_name in row.ops},
        selected_backends={},
    )

    spec = row.spec
    if spec is None:
        yield stats
        return

    previous_backend = os.environ.get("RACETRACK_KERNEL_BACKEND")
    os.environ["RACETRACK_KERNEL_BACKEND"] = row.backend

    dispatcher = KernelDispatcher(row.kernel_root or spec.kernel_root)
    disabled = REAL_DISABLED_BACKENDS.get(row.partition_model, frozenset())
    if disabled:
        dispatcher.BACKENDS = tuple(
            b for b in dispatcher.BACKENDS if b not in disabled
        )
        dispatcher._best_fast_path = {
            op_name: backend
            for op_name, backend in dispatcher._best_fast_path.items()
            if backend not in disabled
        }

    from racetrack.pre_trace import apply_pre_trace_patches, _set_attr
    originals = apply_pre_trace_patches(model, spec, dispatcher)

    _attach_kernel_dispatcher_for_fx_ops(model, spec, dispatcher, stats, originals)

    try:
        yield stats
        if row.backend == "best":
            selected = {}
            for op_name, backends in dispatcher._best_ops.items():
                selected[op_name] = tuple(sorted(backends))
            for op_name in row.ops:
                if op_name not in selected:
                    cached = dispatcher._best_fast_path.get(op_name)
                    if cached and cached != "torch":
                        selected[op_name] = (cached,)
                    else:
                        for b in CONCRETE_BACKENDS:
                            if dispatcher._resolve(b, op_name) is not None:
                                selected[op_name] = (b,)
                                break
                        # Ops with no concrete kernel are omitted (use fallback)
            stats.selected_backends.update(selected)
        else:
            stats.selected_backends.update(
                {op_name: (row.backend,) for op_name in row.ops}
            )
        if strict_kernel_use and not row.ops:
            raise RuntimeError(f"{row.label} has no ops")
    finally:
        rollback_patches(originals)
        if previous_backend is None:
            os.environ.pop("RACETRACK_KERNEL_BACKEND", None)
        else:
            os.environ["RACETRACK_KERNEL_BACKEND"] = previous_backend


def _attach_kernel_dispatcher_for_fx_ops(
    model: torch.nn.Module,
    spec: PartitionSpec,
    dispatcher: KernelDispatcher,
    stats: PatchStats,
    originals: Originals,
) -> None:
    """For fx_pattern ops in eager mode, attach kernel_dispatcher to modules.

    The inference model's _kernel_call() checks for kernel_dispatcher on
    modules and routes through it. This handles fused_swiglu on MLP/Expert
    and fused_residual_norm on RMSNorm when NOT using torch.compile.
    """
    from inference import model as rm
    from racetrack.pre_trace import _set_attr

    fx_op_names = {op.name for op in spec.fx_ops}
    if not fx_op_names:
        return

    for module in model.modules():
        attach = False
        if "fused_residual_norm" in fx_op_names and isinstance(module, rm.RMSNorm):
            attach = True
        if "fused_swiglu" in fx_op_names and isinstance(module, (rm.MLP, rm.Expert)):
            attach = True
        if attach:
            _set_attr(module, "kernel_dispatcher", _make_stats_dispatcher(dispatcher, stats, spec), originals)
            _set_attr(module, "kernel_stats", stats, originals)


class _StatsDispatcher:
    """Wraps KernelDispatcher to track call counts for PatchStats."""

    def __init__(self, dispatcher: KernelDispatcher, stats: PatchStats, spec: PartitionSpec):
        self._dispatcher = dispatcher
        self._stats = stats
        self._spec = spec

    @property
    def _best_ops(self):
        return self._dispatcher._best_ops

    def call(self, op_name, fallback, *args, **kwargs):
        if self._spec.model == "dsv3_2":
            return self._call_dsv3_2(op_name, fallback, *args, **kwargs)
        return self._dispatcher.call(op_name, fallback, *args, **kwargs)

    def _call_dsv3_2(self, op_name, fallback, *args, **kwargs):
        """Handle dsv3_2 legacy contract (2D flat, residual-first arg order)."""
        if op_name == "fused_residual_norm":
            return self._call_residual_norm_dsv3_2(fallback, *args, **kwargs)
        if op_name == "fused_swiglu":
            return self._call_swiglu_dsv3_2(fallback, *args, **kwargs)
        return self._dispatcher.call(op_name, fallback, *args, **kwargs)

    def _call_residual_norm_dsv3_2(self, fallback, update, residual, weight, *, eps):
        shape = update.shape
        cols = shape[-1]
        update_flat = update.contiguous().view(-1, cols)
        residual_flat = residual.contiguous().view(-1, cols)

        def legacy_fallback(r, u, w, *, eps):
            normed, hidden = fallback(u.view(shape), r.view(shape), w, eps=eps)
            return hidden.contiguous().view(-1, cols), normed.contiguous().view(-1, cols)

        hidden, normed = self._dispatcher.call(
            "fused_residual_norm", legacy_fallback,
            residual_flat, update_flat, weight, eps=eps,
        )
        return normed.view(shape), hidden.view(shape)

    def _call_swiglu_dsv3_2(self, fallback, gate, up):
        shape = gate.shape
        cols = shape[-1]
        gate_flat = gate.contiguous().view(-1, cols)
        up_flat = up.contiguous().view(-1, cols)

        def legacy_fallback(g, u):
            return fallback(g.view(shape), u.view(shape)).contiguous().view(-1, cols)

        return self._dispatcher.call(
            "fused_swiglu", legacy_fallback, gate_flat, up_flat,
        ).view(shape)


def _make_stats_dispatcher(dispatcher, stats, spec):
    return _StatsDispatcher(dispatcher, stats, spec)
