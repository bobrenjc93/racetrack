"""Partition spec data model: defines what a partition fuses and how.

A partition is a declarative spec (spec.py) + hand-written kernels
(kernels/<backend>/<op>.py). The spec drives a torch.compile custom
backend that traces the model, applies pattern-matched replacements
with custom kernel calls, and returns the rewritten graph for
execution (optionally via CUDA graph).
"""
from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class FusedOp:
    name: str
    kind: str  # "fx_pattern", "pre_trace", or "module_patch"

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "kind": self.kind}


@dataclass(frozen=True)
class PartitionSpec:
    model: str
    partition_hash: str
    notes: str
    graph_nodes: dict[str, list[str]]
    fused_ops: tuple[FusedOp, ...]

    @property
    def partition_id(self) -> str:
        # The directory name <hash> is the first 8 hex chars of SHA-256 of the
        # spec.py file bytes (see scripts/gen_partition.py). Hashing only
        # model+fused_ops here would collide across partitions that share
        # FUSED_OPS but differ in GRAPH_NODES/PARTITION_NOTES, so we hash the
        # actual spec.py content to faithfully reproduce the directory name.
        spec_path = self.partition_dir / "spec.py"
        if not spec_path.exists():
            # Synthetic specs with no on-disk spec.py (e.g. the in-memory
            # BASELINE_SPEC) cannot reproduce a directory hash, so fall back to a
            # deterministic id over the in-memory fields rather than raising.
            canonical = repr((self.model, self.partition_hash, self.notes,
                              self.graph_nodes, self.fused_ops))
            return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
        spec_bytes = spec_path.read_bytes()
        return hashlib.sha256(spec_bytes).hexdigest()[:8]

    @property
    def partition_dir(self) -> Path:
        return PROJECT_ROOT / "partitions" / self.model / self.partition_hash

    @property
    def kernel_root(self) -> Path:
        return self.partition_dir / "kernels"

    @property
    def fx_ops(self) -> list[FusedOp]:
        return [op for op in self.fused_ops if op.kind == "fx_pattern"]

    @property
    def pre_trace_ops(self) -> list[FusedOp]:
        return [op for op in self.fused_ops if op.kind == "pre_trace"]

    @property
    def module_patch_ops(self) -> list[FusedOp]:
        return [op for op in self.fused_ops if op.kind == "module_patch"]

    @property
    def op_names(self) -> tuple[str, ...]:
        return tuple(op.name for op in self.fused_ops)

    @property
    def label(self) -> str:
        return f"{self.model}/{self.partition_hash}"


def load_spec(model: str, partition_hash: str) -> PartitionSpec:
    spec_path = PROJECT_ROOT / "partitions" / model / partition_hash / "spec.py"
    if not spec_path.exists():
        raise FileNotFoundError(f"No spec.py at {spec_path}")
    spec_mod = _load_module(spec_path)
    return _spec_from_module(spec_mod, model, partition_hash)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"spec_{path.parent.name}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spec_from_module(mod: Any, model: str, partition_hash: str) -> PartitionSpec:
    notes = getattr(mod, "PARTITION_NOTES", "")
    graph_nodes = getattr(mod, "GRAPH_NODES", {})
    raw_ops = getattr(mod, "FUSED_OPS", [])
    fused_ops = tuple(FusedOp(name=op["name"], kind=op["kind"]) for op in raw_ops)
    return PartitionSpec(
        model=model,
        partition_hash=partition_hash,
        notes=notes,
        graph_nodes=graph_nodes,
        fused_ops=fused_ops,
    )


def discover_partitions(model: str) -> list[PartitionSpec]:
    partitions_dir = PROJECT_ROOT / "partitions" / model
    if not partitions_dir.is_dir():
        return []
    specs = []
    for d in sorted(partitions_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("__"):
            continue
        spec_path = d / "spec.py"
        if not spec_path.exists():
            continue
        if not (d / "kernels").is_dir():
            continue
        try:
            specs.append(load_spec(model, d.name))
        except Exception:
            continue
    return specs


BASELINE_SPEC = PartitionSpec(
    model="baseline",
    partition_hash="baseline",
    notes="No partition — pure torch fallback",
    graph_nodes={},
    fused_ops=(),
)
