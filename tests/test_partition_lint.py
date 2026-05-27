"""Structural lint tests for partition specs and benchmark results.

These tests don't run any kernels — they just validate that the partition
directory structure is consistent and that benchmark results are sane.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from racetrack.partition_spec import PROJECT_ROOT, discover_partitions, load_spec
from racetrack.bench import CONCRETE_KERNEL_BACKENDS

MODELS = ("dsv3_2", "dsv3_2_nvfp4")
BACKENDS = CONCRETE_KERNEL_BACKENDS


def _all_specs():
    specs = []
    for model in MODELS:
        for spec in discover_partitions(model):
            specs.append(spec)
    return specs


def _spec_ids(specs):
    return [f"{s.model}/{s.partition_hash}" for s in specs]


ALL_SPECS = _all_specs()


def _exported_op_names(kernel_file: Path) -> list[str]:
    """Import a kernel file and return all callable names starting with 'fused_'."""
    spec_name = f"_lint_{abs(hash(str(kernel_file)))}"
    spec = importlib.util.spec_from_file_location(spec_name, str(kernel_file))
    if spec is None or spec.loader is None:
        return []
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return []
    return [
        name for name in dir(mod)
        if name.startswith("fused_") and callable(getattr(mod, name))
    ]


# ---------------------------------------------------------------------------
# Test 1: every spec op has a kernel implementation in all 3 backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", ALL_SPECS, ids=_spec_ids(ALL_SPECS))
def test_all_spec_ops_have_all_backends(spec):
    """Every op in FUSED_OPS must have a matching callable in all 3 backend dirs."""
    kernel_root = spec.kernel_root
    missing = []

    for op in spec.fused_ops:
        for backend in BACKENDS:
            backend_dir = kernel_root / backend
            if not backend_dir.is_dir():
                missing.append((op.name, backend, "no backend dir"))
                continue
            found = False
            for py in sorted(backend_dir.glob("*.py")):
                if py.name.startswith("_"):
                    continue
                exported = _exported_op_names(py)
                if op.name in exported:
                    found = True
                    break
            if not found:
                available = set()
                for py in sorted(backend_dir.glob("*.py")):
                    if py.name.startswith("_"):
                        continue
                    available.update(_exported_op_names(py))
                similar = [n for n in available if n.startswith("fused_")]
                hint = f" (available: {', '.join(sorted(similar))})" if similar else ""
                missing.append((op.name, backend, f"no callable{hint}"))

    if missing:
        lines = [f"  {op} / {backend}: {reason}" for op, backend, reason in missing]
        pytest.fail(
            f"{spec.label} has {len(missing)} missing kernel(s):\n" + "\n".join(lines)
        )


# ---------------------------------------------------------------------------
# Test 2: 'best' backend results >= any single concrete backend
# ---------------------------------------------------------------------------

def _parse_leaderboard(md_path: Path) -> list[dict]:
    """Parse the markdown leaderboard table into a list of row dicts."""
    text = md_path.read_text()
    rows = []
    in_table = False
    for line in text.splitlines():
        if "| # |" in line:
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c]
            if len(cells) < 5:
                continue
            partition = cells[1]
            backend_raw = cells[2]
            total_ms = float(cells[3])
            speedup_str = cells[4].replace("x", "")
            speedup = float(speedup_str)
            backend_name = backend_raw.split("(")[0].strip()
            rows.append({
                "partition": partition,
                "backend_raw": backend_raw,
                "backend": backend_name,
                "total_ms": total_ms,
                "speedup": speedup,
            })
        elif in_table and not line.startswith("|"):
            in_table = False
    return rows


def _find_result_files() -> list[Path]:
    results = []
    for md in sorted(PROJECT_ROOT.glob("benchmarks/*/results/*/*.md")):
        results.append(md)
    return results


RESULT_FILES = _find_result_files()


@pytest.mark.parametrize("md_path", RESULT_FILES, ids=[str(p.relative_to(PROJECT_ROOT)) for p in RESULT_FILES])
def test_best_backend_not_slower_than_concrete(md_path):
    """For each partition, the 'best' row must have total_ms <= min(concrete backends).

    'best' picks the fastest kernel per-op, so it should never be slower
    than running a single backend for all ops. We allow 5% tolerance for
    measurement noise.
    """
    rows = _parse_leaderboard(md_path)
    if not rows:
        pytest.skip("No leaderboard rows found")

    partitions = {r["partition"] for r in rows}
    violations = []
    tolerance = 0.05

    for partition in sorted(partitions):
        if partition == "baseline":
            continue
        best_rows = [r for r in rows if r["partition"] == partition and r["backend"] == "best"]
        concrete_rows = [
            r for r in rows
            if r["partition"] == partition and r["backend"] in BACKENDS
        ]
        if not best_rows or not concrete_rows:
            continue

        best_ms = min(r["total_ms"] for r in best_rows)
        for cr in concrete_rows:
            allowed = cr["total_ms"] * (1 + tolerance)
            if best_ms > allowed:
                pct_slower = (best_ms / cr["total_ms"] - 1) * 100
                violations.append(
                    f"  {partition}: best={best_ms:.1f}ms > "
                    f"{cr['backend']}={cr['total_ms']:.1f}ms "
                    f"(+{pct_slower:.1f}% slower, tolerance={tolerance*100:.0f}%)"
                )

    if violations:
        pytest.fail(
            f"{md_path.name} has {len(violations)} 'best' violation(s):\n"
            + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# Test 3: every non-baseline partition has all 3 backends in the leaderboard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("md_path", RESULT_FILES, ids=[str(p.relative_to(PROJECT_ROOT)) for p in RESULT_FILES])
def test_all_partitions_have_all_backends(md_path):
    """Every non-baseline partition in the leaderboard must have a row for
    each of the three concrete backends (triton, cutedsl, helion) plus best.
    A missing backend means the benchmark runner skipped or crashed on it.
    """
    rows = _parse_leaderboard(md_path)
    if not rows:
        pytest.skip("No leaderboard rows found")

    expected = set(BACKENDS) | {"best"}
    partitions = {r["partition"] for r in rows if r["partition"] != "baseline"}
    missing = []

    for partition in sorted(partitions):
        present = {r["backend"] for r in rows if r["partition"] == partition}
        for backend in sorted(expected - present):
            missing.append(f"  {partition}/{backend}")

    if missing:
        pytest.fail(
            f"{md_path.name} is missing {len(missing)} partition/backend row(s):\n"
            + "\n".join(missing)
        )
