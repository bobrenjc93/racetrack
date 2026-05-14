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
BACKENDS = ("torch", "triton", "cutedsl", "helion")

CASES: list[tuple[str, int]] = [
    ("prefill_512", 512),
    ("prefill_2048", 2048),
    ("prefill_4096", 4096),
]

MODEL_OVERRIDES: dict[str, int | float | str] = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "head_dim": 128,
    "q_lora_rank": 1024,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "moe_intermediate_size": 2048,
    "num_layers": 4,
}


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


def _discover_kernel_map(dispatcher, backend: str) -> dict[str, str] | None:
    kernel_map: dict[str, str] = {}
    for mod in dispatcher._load_backend_modules(backend):
        if not bool(getattr(mod, "BACKEND_AVAILABLE", False)):
            continue
        for name in dir(mod):
            if name.startswith("_") or name == "BACKEND_AVAILABLE":
                continue
            if callable(getattr(mod, name)):
                kernel_map[name] = backend
    return kernel_map or None


def run(
    device_str: str = "cuda:0",
    dtype_str: str = "auto",
    warmup: int = 10,
    repeat: int = 30,
    partition_filter: str = "all",
    backend_filter: str = "all",
) -> list[Result]:
    device = torch.device(device_str)
    dtype = _resolve_dtype(dtype_str, device)

    results: list[Result] = []
    for model_name in MODELS:
        all_partitions = _discover_partitions(model_name)
        if partition_filter != "all":
            all_partitions = [p for p in all_partitions if p in partition_filter.split(",")]
            if not all_partitions:
                raise KeyError(f"No partitions matched filter {partition_filter!r}")
        for partition in all_partitions:
            if backend_filter == "all":
                backends = ["torch"] if partition == "baseline" else list(BACKENDS)
            else:
                backends = backend_filter.split(",")
            for backend in backends:
                os.environ["RACETRACK_KERNEL_BACKEND"] = backend
                module = _load_partition_module(model_name, partition)
                model = module.build_model(**MODEL_OVERRIDES).to(device=device, dtype=dtype).eval()

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
                    if dispatcher is not None and backend != "torch":
                        kernel_map = _discover_kernel_map(dispatcher, backend)

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
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
        info["cuda"] = torch.version.cuda or "unknown"
    return info


def _hardware_slug(device: str) -> str:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "cpu"
    idx = int(device.split(":")[-1]) if ":" in device else 0
    props = torch.cuda.get_device_properties(idx)
    gpu_count = torch.cuda.device_count()
    name = props.name.lower()
    for prefix in ("nvidia ", "amd ", "intel "):
        name = name.removeprefix(prefix)
    slug = name.replace(" ", "_")
    return f"{gpu_count}x{slug}"


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


def _discover_dispatched_ops(results: list[Result], partition: str) -> list[str]:
    for r in results:
        if r.partition != partition or r.backend == "torch":
            continue
        module = _load_partition_module(r.model, partition)
        model = module.build_model(**MODEL_OVERRIDES)
        dispatcher = getattr(model, "dispatcher", None)
        if dispatcher is None:
            return []
        ops: list[str] = []
        for backend in ("triton", "helion", "cutedsl"):
            for mod in dispatcher._load_backend_modules(backend):
                if not bool(getattr(mod, "BACKEND_AVAILABLE", False)):
                    continue
                for name in sorted(dir(mod)):
                    if not name.startswith("_") and name != "BACKEND_AVAILABLE" and callable(getattr(mod, name)):
                        if name not in ops:
                            ops.append(name)
        del model
        return ops
    return []


def _synthesize_best(results: list[Result]) -> list[Result]:
    """For each partition, pick the backend with the lowest aggregate time."""
    by_partition: dict[str, dict[str, list[Result]]] = {}
    for r in results:
        by_partition.setdefault(r.partition, {}).setdefault(r.backend, []).append(r)

    best_results: list[Result] = []
    for partition, backends in by_partition.items():
        if partition == "baseline":
            continue
        best_backend = min(
            backends.items(),
            key=lambda kv: sum(r.mean_ms for r in kv[1]),
        )
        backend_name, runs = best_backend
        dispatched_ops = _discover_dispatched_ops(results, partition)
        kernel_map = {op: backend_name for op in dispatched_ops} if dispatched_ops else None
        for r in runs:
            best_results.append(Result(
                model=r.model,
                partition=r.partition,
                backend="best",
                backend_status=f"={backend_name}",
                case=r.case,
                tokens=r.tokens,
                device=r.device,
                dtype=r.dtype,
                mean_ms=r.mean_ms,
                min_ms=r.min_ms,
                max_ms=r.max_ms,
                tokens_per_second=r.tokens_per_second,
                kernels=kernel_map,
            ))
    return best_results


def pick_winner(results: list[Result]) -> dict:
    all_results = results + _synthesize_best(results)

    combos: dict[tuple, list[Result]] = {}
    for r in all_results:
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


def _render_markdown(report: dict, slug: str) -> str:
    hw = report["hardware"]
    winner = report["winner"]
    eval_result = report.get("eval")
    lines: list[str] = []

    lines.append(f"# GSM8K Benchmark: {slug}")
    lines.append("")
    lines.append(f"**GPU**: {hw.get('gpu', 'N/A')} x{hw.get('gpu_count', 1)}  ")
    lines.append(f"**CUDA**: {hw.get('cuda', 'N/A')}  ")
    lines.append(f"**PyTorch**: {hw.get('torch', 'N/A')}  ")
    lines.append(f"**dtype**: {report.get('dtype', 'N/A')}  ")
    lines.append(f"**Date**: {report.get('timestamp', 'N/A')}")
    if eval_result:
        lines.append(f"**Eval model**: {eval_result['model']}  ")
        lines.append(
            f"**GSM8K accuracy**: {eval_result['accuracy_pct']}% "
            f"({eval_result['correct']}/{eval_result['num_samples']})"
        )
    lines.append("")

    lines.append("## Winner")
    lines.append("")
    speedup = winner.get("speedup_vs_baseline")
    speedup_str = f" ({speedup:.3f}x vs baseline)" if speedup is not None else ""
    lines.append(f"**{winner['partition']}/{winner['backend']}**{speedup_str}  ")
    lines.append(f"Aggregate: {winner['aggregate_mean_ms']:.1f}ms")
    if winner.get("kernels"):
        lines.append("")
        lines.append("Kernel dispatch:")
        for op, backend in sorted(winner["kernels"].items()):
            lines.append(f"- `{op}` -> {backend}")
    lines.append("")

    lines.append("## Leaderboard")
    lines.append("")
    headers = ["#", "partition", "backend", "total (ms)", "vs baseline"]
    if eval_result:
        headers.append("correctness")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for rank, entry in enumerate(report["leaderboard"], 1):
        speedup_val = entry.get("speedup_vs_baseline")
        speedup_cell = f"{speedup_val:.3f}x" if speedup_val is not None else "-"
        kernels_note = ""
        if entry["backend"] == "best" and entry.get("kernels"):
            kernels_note = " (" + ", ".join(
                f"{op}={b}" for op, b in sorted(entry["kernels"].items())
            ) + ")"
        row = [
            str(rank),
            entry["partition"],
            f"{entry['backend']}{kernels_note}",
            f"{entry['aggregate_mean_ms']:.1f}",
            speedup_cell,
        ]
        if eval_result:
            row.append(f"{eval_result['accuracy_pct']}%")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if report["leaderboard"]:
        first_cases = report["leaderboard"][0]["cases"]
        lines.append("## Cases")
        lines.append("")
        for case_name, case_data in first_cases.items():
            lines.append(f"- **{case_name}**: {case_data['tokens']} tokens")
        lines.append("")

    baseline = report.get("baseline", {})
    if baseline:
        lines.append("## Baseline reference")
        lines.append("")
        for case_name, ms in baseline.items():
            lines.append(f"- {case_name}: {ms:.1f}ms")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K-shaped benchmark")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--partition", default="all", help="all, baseline, or comma-separated hashes")
    parser.add_argument("--backend", default="all", help="all, or comma-separated: torch,triton,cutedsl,helion")
    parser.add_argument("--no-eval", action="store_true", help="Skip GSM8K accuracy eval")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    eval_result = None
    if not args.no_eval:
        from benchmarks.gsm8k.eval import CACHE_PATH

        if CACHE_PATH.exists():
            import json as _json
            _cache = _json.loads(CACHE_PATH.read_text())
            if _cache:
                eval_result = next(iter(_cache.values()))
                print(f"Loaded cached eval: {eval_result['accuracy_pct']}% "
                      f"({eval_result['correct']}/{eval_result['num_samples']})")
        if eval_result is None:
            print("No cached eval results. Run the eval first with torchrun:\n"
                  "  torchrun --standalone --nproc-per-node=8 \\\n"
                  "    -m benchmarks.gsm8k.eval \\\n"
                  "    --ckpt-path checkpoints/dsv3_2-mp8")

    results = run(args.device, args.dtype, args.warmup, args.repeat, args.partition, args.backend)
    _print_table(results)

    report = pick_winner(results)
    if eval_result is not None:
        report["eval"] = eval_result
    winner = report["winner"]
    slug = _hardware_slug(args.device)
    results_dir = BENCHMARK_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"{slug}.md"
    md = _render_markdown(report, slug)
    results_path.write_text(md)
    speedup = winner.get("speedup_vs_baseline")
    speedup_str = f" ({speedup:.3f}x vs baseline)" if speedup is not None else ""
    print(f"\nWinner: {winner['partition']}/{winner['backend']} "
          f"({winner['aggregate_mean_ms']:.1f}ms aggregate){speedup_str}")
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    main()
