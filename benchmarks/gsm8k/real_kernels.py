"""Patch real DeepSeek inference modules with partition-local kernels.

The synthetic partition models own their own randomly initialized parameters.
For real GSM8K benchmarking we instead keep the checkpoint-backed
``inference.model.Transformer`` intact and patch compatible primitive ops in
place. A leaderboard row is therefore:

    real full model + one partition kernel directory + one backend

Only ops with the same tensor contract as the real inference modules are
patched here. Unsupported partition fusions are left out of the real
leaderboard until they have a real-model adapter.
"""

from __future__ import annotations

import contextlib
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
import torch.nn.functional as F

from racetrack.runtime.dispatch import KernelDispatcher


SUPPORTED_OPS = frozenset({"fused_residual_norm", "fused_swiglu"})
BACKENDS = ("torch", "triton", "cutedsl", "helion", "best")


@dataclass(frozen=True)
class RealKernelRow:
    partition_model: str
    partition: str
    backend: str
    kernel_root: Path | None
    ops: tuple[str, ...]

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
        )
    ]
    root = Path(__file__).resolve().parents[2] / "partitions" / partition_model
    partitions = sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "kernels").is_dir() and not p.name.startswith("__")
    )
    if partition_filter != "all":
        wanted = {p.strip() for p in partition_filter.split(",") if p.strip()}
        partitions = [p for p in partitions if p.name in wanted]
        if not partitions:
            raise KeyError(f"No partitions matched filter {partition_filter!r}")

    for partition_dir in partitions:
        kernel_root = partition_dir / "kernels"
        backends = _discover_backends(kernel_root)
        if backend_filter != "all":
            wanted_backends = {
                _normalize_backend_name(b)
                for b in backend_filter.split(",")
                if b.strip()
            }
            backends = [b for b in backends if b in wanted_backends]
        for backend in backends:
            ops = tuple(sorted(_discover_supported_ops(kernel_root, backend)))
            if not ops and backend != "best":
                continue
            if backend == "best":
                ops = tuple(sorted(_discover_supported_ops(kernel_root, "triton")
                                   | _discover_supported_ops(kernel_root, "cutedsl")
                                   | _discover_supported_ops(kernel_root, "helion")))
            if not ops:
                continue
            rows.append(
                RealKernelRow(
                    partition_model=partition_model,
                    partition=partition_dir.name,
                    backend=backend,
                    kernel_root=kernel_root,
                    ops=ops,
                )
            )
    return rows


def _normalize_backend_name(backend: str) -> str:
    backend = backend.strip().lower()
    if backend == "cutedl":
        return "cutedsl"
    if backend == "torch_compile":
        return "torch.compile"
    return backend


def _discover_backends(kernel_root: Path) -> list[str]:
    backends = [
        backend for backend in ("triton", "cutedsl", "helion")
        if (kernel_root / backend).is_dir()
    ]
    if len(backends) > 1:
        backends.append("best")
    return backends


def _discover_supported_ops(kernel_root: Path, backend: str) -> set[str]:
    if backend == "torch":
        return set()
    dispatcher = KernelDispatcher(kernel_root)
    ops = set()
    for op_name in SUPPORTED_OPS:
        if dispatcher._resolve(backend, op_name) is not None:
            ops.add(op_name)
    return ops


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

    dispatcher = KernelDispatcher(row.kernel_root)
    stats = PatchStats(
        calls={op_name: 0 for op_name in row.ops},
        selected_backends={},
    )
    originals: list[tuple[torch.nn.Module, str, Callable]] = []
    previous_backend = os.environ.get("RACETRACK_KERNEL_BACKEND")
    os.environ["RACETRACK_KERNEL_BACKEND"] = row.backend

    try:
        _patch_modules(model, row, dispatcher, stats, originals)
        yield stats
        if row.backend == "best":
            stats.selected_backends.update(
                {
                    op_name: tuple(sorted(backends))
                    for op_name, backends in dispatcher._best_ops.items()
                }
            )
        else:
            stats.selected_backends.update(
                {op_name: (row.backend,) for op_name, calls in stats.calls.items() if calls}
            )
        if strict_kernel_use and stats.total_calls == 0:
            raise RuntimeError(f"{row.label} did not exercise any compatible real-model kernels")
        if strict_kernel_use and not stats.used_partition_kernel:
            raise RuntimeError(f"{row.label} did not use any non-torch partition kernel")
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)
        if previous_backend is None:
            os.environ.pop("RACETRACK_KERNEL_BACKEND", None)
        else:
            os.environ["RACETRACK_KERNEL_BACKEND"] = previous_backend


def _patch_modules(
    model: torch.nn.Module,
    row: RealKernelRow,
    dispatcher: KernelDispatcher,
    stats: PatchStats,
    originals: list[tuple[torch.nn.Module, str, Callable]],
) -> None:
    from inference import model as real_model

    for module in model.modules():
        if "fused_residual_norm" in row.ops and isinstance(module, real_model.RMSNorm):
            original = module.forward
            originals.append((module, "forward", original))
            module.forward = types.MethodType(
                _make_rms_norm_forward(original, dispatcher, stats),
                module,
            )
        if "fused_swiglu" in row.ops and isinstance(module, (real_model.MLP, real_model.Expert)):
            original = module.forward
            originals.append((module, "forward", original))
            module.forward = types.MethodType(
                _make_swiglu_forward(dispatcher, stats),
                module,
            )


def _make_rms_norm_forward(
    original: Callable,
    dispatcher: KernelDispatcher,
    stats: PatchStats,
) -> Callable:
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return original(x, residual)

        def fallback(
            update: torch.Tensor,
            prev_residual: torch.Tensor,
            weight: torch.Tensor,
            *,
            eps: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = update.float() + prev_residual.float()
            var = hidden.pow(2).mean(-1, keepdim=True)
            normed = hidden * torch.rsqrt(var + eps)
            return (weight * normed).to(update.dtype), hidden.to(update.dtype)

        stats.calls["fused_residual_norm"] = stats.calls.get("fused_residual_norm", 0) + 1
        return dispatcher.call(
            "fused_residual_norm",
            fallback,
            x,
            residual,
            self.weight,
            eps=self.eps,
        )

    return forward


def _make_swiglu_forward(
    dispatcher: KernelDispatcher,
    stats: PatchStats,
) -> Callable:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        up = self.w3(x)

        def fallback(gate_values: torch.Tensor, up_values: torch.Tensor) -> torch.Tensor:
            return (F.silu(gate_values.float()) * up_values.float()).type_as(gate_values)

        stats.calls["fused_swiglu"] = stats.calls.get("fused_swiglu", 0) + 1
        hidden = dispatcher.call("fused_swiglu", fallback, gate, up)
        return self.w2(hidden.type_as(x))

    return forward
