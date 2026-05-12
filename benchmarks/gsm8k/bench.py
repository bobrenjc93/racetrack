"""GSM8K-shaped benchmark.

Measures throughput at sequence lengths representative of the GSM8K dataset:
  - question:  96 tokens  (median question length)
  - cot:      256 tokens  (chain-of-thought answer)
  - full:     384 tokens  (question + full answer)

Sweeps all partitions and kernel backends, picks the fastest combination,
and writes the result to winner.json alongside this file.

Usage:
    python -m benchmarks.gsm8k.bench
    python -m benchmarks.gsm8k.bench --device cpu --dtype float32
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

BENCHMARK_DIR = Path(__file__).parent
MODELS = ("dsv3_2",)
BACKENDS = ("torch", "triton", "helion", "best")

CASES: list[tuple[str, int]] = [
    ("question", 96),
    ("cot", 256),
    ("full", 384),
]


@dataclass
class Result:
    model: str
    partition: str
    backend: str
    backend_status: str
    case: str
    tokens: int
    device: str
    dtype: str
    mean_ms: float
    min_ms: float
    max_ms: float
    tokens_per_second: float
    kernels: dict[str, str] | None = None


def _load_partition_module(model_name: str, partition: str):
    if partition == "baseline":
        return importlib.import_module(f"partitions.{model_name}.model")
    return importlib.import_module(f"partitions.{model_name}.{partition}.model")


def _discover_partitions(model_name: str) -> list[str]:
    root = Path(__file__).resolve().parents[2] / "partitions" / model_name
    partitions = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "model.py").exists() and not p.name.startswith("__")
    )
    return ["baseline", *partitions]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> list[float]:
    for _ in range(warmup):
        model(input_ids, positions)
    _sync(device)

    times: list[float] = []
    for _ in range(repeat):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(input_ids, positions)
            end.record()
            torch.cuda.synchronize(device)
            times.append(float(start.elapsed_time(end)))
        else:
            t0 = time.perf_counter()
            model(input_ids, positions)
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise KeyError(f"Unknown dtype {name!r}")
    return mapping[name]


def run(
    device_str: str = "cuda:0",
    dtype_str: str = "auto",
    warmup: int = 3,
    repeat: int = 10,
) -> list[Result]:
    device = torch.device(device_str)
    dtype = _resolve_dtype(dtype_str, device)

    results: list[Result] = []
    for model_name in MODELS:
        for partition in _discover_partitions(model_name):
            backends = ["torch"] if partition == "baseline" else list(BACKENDS)
            for backend in backends:
                os.environ["RACETRACK_KERNEL_BACKEND"] = backend
                module = _load_partition_module(model_name, partition)
                model = module.build_model().to(device=device, dtype=dtype).eval()

                dispatcher = getattr(model, "dispatcher", None)

                for case_name, tokens in CASES:
                    input_ids = torch.arange(tokens, device=device, dtype=torch.long) % 4096
                    positions = torch.arange(tokens, device=device, dtype=torch.long)

                    times = _time_forward(
                        model, input_ids, positions,
                        warmup=warmup, repeat=repeat, device=device,
                    )
                    mean_ms = sum(times) / len(times)

                    status = "native"
                    kernel_map = None
                    if dispatcher is not None:
                        kernel_map = dict(dispatcher._best_fast_path) or None
                        if hasattr(dispatcher, "best_summary"):
                            status = dispatcher.best_summary()

                    results.append(Result(
                        model=model_name,
                        partition=partition,
                        backend=backend,
                        backend_status=status,
                        case=case_name,
                        tokens=tokens,
                        device=str(device),
                        dtype=str(dtype).replace("torch.", ""),
                        mean_ms=mean_ms,
                        min_ms=min(times),
                        max_ms=max(times),
                        tokens_per_second=tokens / (mean_ms / 1000.0),
                        kernels=kernel_map,
                    ))

                del model
                _sync(device)

    return results


def _combo_key(r: Result) -> tuple[str, str, str]:
    return (r.model, r.partition, r.backend)


def _hardware_info(device: str) -> dict:
    info: dict = {
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = int(device.split(":")[-1]) if ":" in device else 0
        props = torch.cuda.get_device_properties(idx)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
        info["cuda"] = torch.version.cuda or "unknown"
    return info


def _build_combo_entry(
    runs: list[Result],
    baseline_ms: dict[str, float],
) -> dict:
    kernel_map = next((r.kernels for r in runs if r.kernels), None)
    aggregate = sum(r.mean_ms for r in runs)
    baseline_aggregate = sum(baseline_ms.get(r.case, r.mean_ms) for r in runs)
    return {
        "partition": runs[0].partition,
        "backend": runs[0].backend,
        "backend_status": runs[0].backend_status,
        "kernels": kernel_map,
        "aggregate_mean_ms": round(aggregate, 3),
        "speedup_vs_baseline": round(baseline_aggregate / aggregate, 4) if aggregate > 0 else None,
        "cases": {
            r.case: {
                "tokens": r.tokens,
                "mean_ms": round(r.mean_ms, 3),
                "min_ms": round(r.min_ms, 3),
                "max_ms": round(r.max_ms, 3),
                "tokens_per_second": round(r.tokens_per_second, 1),
                "vs_baseline": round(
                    baseline_ms.get(r.case, r.mean_ms) / r.mean_ms, 4
                ) if r.mean_ms > 0 else None,
            }
            for r in runs
        },
    }


def pick_winner(results: list[Result]) -> dict:
    combos: dict[tuple, list[Result]] = {}
    for r in results:
        combos.setdefault(_combo_key(r), []).append(r)

    baseline_key = next(
        (k for k in combos if k[1] == "baseline" and k[2] == "torch"), None
    )
    baseline_ms: dict[str, float] = {}
    if baseline_key is not None:
        for r in combos[baseline_key]:
            baseline_ms[r.case] = r.mean_ms

    ranked = sorted(
        combos.items(),
        key=lambda kv: sum(r.mean_ms for r in kv[1]),
    )

    winner_key, winner_runs = ranked[0]
    winner_entry = _build_combo_entry(winner_runs, baseline_ms)

    leaderboard = []
    for key, runs in ranked:
        leaderboard.append(_build_combo_entry(runs, baseline_ms))

    return {
        "winner": winner_entry,
        "baseline": {
            case: round(ms, 3) for case, ms in baseline_ms.items()
        },
        "leaderboard": leaderboard,
        "model": winner_runs[0].model,
        "dtype": winner_runs[0].dtype,
        "hardware": _hardware_info(winner_runs[0].device),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _print_table(results: list[Result]) -> None:
    headers = ["model", "partition", "backend", "status", "case", "tokens",
               "mean_ms", "tok/s"]
    rows = []
    for r in results:
        rows.append([
            r.model, r.partition, r.backend, r.backend_status, r.case,
            str(r.tokens), f"{r.mean_ms:.3f}", f"{r.tokens_per_second:.1f}",
        ])
    widths = [
        max(len(str(row[i])) for row in [headers, *rows])
        for i in range(len(headers))
    ]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K-shaped benchmark")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    results = run(args.device, args.dtype, args.warmup, args.repeat)
    _print_table(results)

    report = pick_winner(results)
    winner = report["winner"]
    winner_path = BENCHMARK_DIR / "winner.json"
    with open(winner_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    speedup = winner.get("speedup_vs_baseline")
    speedup_str = f" ({speedup:.3f}x vs baseline)" if speedup is not None else ""
    print(f"\nWinner: {winner['partition']}/{winner['backend']} "
          f"({winner['aggregate_mean_ms']:.1f}ms aggregate){speedup_str}")
    print(f"Saved to {winner_path}")


if __name__ == "__main__":
    main()
