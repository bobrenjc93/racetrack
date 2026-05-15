"""GSM8K benchmark harness.

Measures throughput at sequence lengths representative of the GSM8K dataset:
  - question:  96 tokens  (median question length)
  - cot:      256 tokens  (chain-of-thought answer)
  - full:     384 tokens  (question + full answer)

Local optimization runs use dummy synthetic weights and validate each row's
logits against the baseline implementation for the same synthetic inputs. A
published GSM8K report must use real weights and a Hugging Face token; this
entry point refuses to publish synthetic results as GSM8K accuracy.

Usage:
    python -m benchmarks.gsm8k.bench --dummy-weights
    python -m benchmarks.gsm8k.bench --dummy-weights --device cpu --dtype float32
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.gsm8k.hf_auth import require_hf_token

BENCHMARK_DIR = Path(__file__).parent
MODELS = ("dsv3_2", "dsv3_2_nvfp4")
TORCH_COMPILE_BACKEND = "torch.compile"
TORCH_COMPILE_ALIASES = {TORCH_COMPILE_BACKEND, "torch_compile"}
CONCRETE_BACKENDS = ("triton", "cutedsl", "helion")
PARTITION_BACKENDS = ("torch", *CONCRETE_BACKENDS)
BASELINE_BACKENDS = ("torch", TORCH_COMPILE_BACKEND)

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

REAL_WEIGHT_UNSUPPORTED = (
    "Real-weight per-row GSM8K benchmark generation is not supported by this "
    "runner yet. The current partition models are single-process synthetic "
    "shape models, while the DeepSeek-V3.2 checkpoint is an 8-way model-parallel "
    "checkpoint and the partition architecture still lacks full checkpoint "
    "coverage for the dense first layers and NVFP4/indexer weights. Use "
    "--dummy-weights for local synthetic optimization; do not publish those "
    "numbers as GSM8K accuracy. Use benchmarks.gsm8k.real_bench for the "
    "checkpoint-backed end-to-end leaderboard."
)


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
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    ok: bool = True
    kernels: dict[str, str] | None = None


def _load_partition_module(model_name: str, partition: str):
    if partition == "baseline":
        return importlib.import_module(f"partitions.{model_name}.model")
    return importlib.import_module(f"partitions.{model_name}.{partition}.model")


def _normalize_backend_name(backend: str) -> str:
    backend = backend.strip().lower()
    if backend in TORCH_COMPILE_ALIASES:
        return TORCH_COMPILE_BACKEND
    if backend == "cutedl":
        return "cutedsl"
    return backend


def _env_backend(backend: str) -> str:
    return "torch" if backend == TORCH_COMPILE_BACKEND else backend


def _compile_model_if_requested(model: torch.nn.Module, backend: str) -> torch.nn.Module:
    if backend != TORCH_COMPILE_BACKEND:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    return torch.compile(model)


def _cleanup_compile_state(device: torch.device) -> None:
    dynamo = getattr(torch, "_dynamo", None)
    reset = getattr(dynamo, "reset", None)
    if callable(reset):
        reset()
    if device.type == "cuda":
        torch.cuda.empty_cache()


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
) -> tuple[torch.Tensor, list[float]]:
    output = None
    for _ in range(warmup):
        output = model(input_ids, positions)
    _sync(device)

    if device.type == "cuda":
        if output is None:
            output = model(input_ids, positions)
            _sync(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(input_ids, positions)

        times: list[float] = []
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize(device)
            times.append(float(start.elapsed_time(end)))
    else:
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            output = model(input_ids, positions)
            times.append((time.perf_counter() - t0) * 1000.0)
    assert output is not None
    return output, times


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


def _validate_output(
    output: torch.Tensor,
    baseline: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float, bool]:
    if output.shape != baseline.shape:
        return math.inf, math.inf, False

    output_float = output.float()
    baseline_float = baseline.float()
    output_finite = torch.isfinite(output_float)
    baseline_finite = torch.isfinite(baseline_float)
    if not torch.equal(output_finite, baseline_finite):
        return math.inf, math.inf, False

    finite_mask = output_finite & baseline_finite
    if bool(finite_mask.any()):
        diff = float((output_float[finite_mask] - baseline_float[finite_mask]).abs().max().item())
        ref_scale = max(float(baseline_float[finite_mask].abs().max().item()), 1.0e-12)
    else:
        diff = 0.0
        ref_scale = 1.0
    rel_diff = diff / ref_scale
    ok = math.isfinite(diff) and diff <= atol + rtol * ref_scale
    return diff, rel_diff, ok


def _benchmark_backend(
    *,
    model_name: str,
    partition: str,
    backend: str,
    case_name: str,
    tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    warmup: int,
    repeat: int,
    baseline_out: torch.Tensor | None,
    atol: float,
    rtol: float,
) -> tuple[Result, torch.Tensor]:
    os.environ["RACETRACK_KERNEL_BACKEND"] = _env_backend(backend)
    module = _load_partition_module(model_name, partition)
    model = module.build_model(**MODEL_OVERRIDES).to(device=device, dtype=dtype).eval()
    model = _compile_model_if_requested(model, backend)

    output, times = _time_forward(
        model, input_ids, positions,
        warmup=warmup, repeat=repeat, device=device,
    )
    mean_ms = sum(times) / len(times)

    if baseline_out is None:
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        ok = True
    else:
        max_abs_diff, max_rel_diff, ok = _validate_output(
            output, baseline_out, atol=atol, rtol=rtol,
        )

    status = "compiled" if backend == TORCH_COMPILE_BACKEND else "native"
    dispatcher = getattr(model, "dispatcher", None)
    kernel_map = None
    if dispatcher is not None and backend in CONCRETE_BACKENDS:
        kernel_map = _discover_kernel_map(dispatcher, backend)

    result = Result(
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
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        ok=ok,
        kernels=kernel_map,
    )

    del model
    _sync(device)
    if backend == TORCH_COMPILE_BACKEND:
        _cleanup_compile_state(device)
        _sync(device)
    return result, output


def run(
    device_str: str = "cuda:0",
    dtype_str: str = "auto",
    warmup: int = 10,
    repeat: int = 30,
    partition_filter: str = "all",
    backend_filter: str = "all",
    atol: float = 5.0e-2,
    rtol: float = 1.0e-2,
) -> list[Result]:
    """Run the synthetic dummy-weight shape benchmark."""
    device = torch.device(device_str)
    dtype = _resolve_dtype(dtype_str, device)

    results: list[Result] = []
    for model_name in MODELS:
        all_partitions = _discover_partitions(model_name)
        if partition_filter != "all":
            all_partitions = [p for p in all_partitions if p in partition_filter.split(",")]
            if not all_partitions:
                raise KeyError(f"No partitions matched filter {partition_filter!r}")
        for case_name, tokens in CASES:
            input_ids = torch.arange(tokens, device=device, dtype=torch.long) % 4096
            positions = torch.arange(tokens, device=device, dtype=torch.long)

            baseline_result, baseline_out = _benchmark_backend(
                model_name=model_name,
                partition="baseline",
                backend="torch",
                case_name=case_name,
                tokens=tokens,
                device=device,
                dtype=dtype,
                input_ids=input_ids,
                positions=positions,
                warmup=warmup,
                repeat=repeat,
                baseline_out=None,
                atol=atol,
                rtol=rtol,
            )

            for partition in all_partitions:
                if backend_filter == "all":
                    backends = (
                        list(BASELINE_BACKENDS)
                        if partition == "baseline"
                        else list(PARTITION_BACKENDS)
                    )
                else:
                    backends = [
                        _normalize_backend_name(backend)
                        for backend in backend_filter.split(",")
                        if backend.strip()
                    ]
                for backend in backends:
                    if partition == "baseline" and backend == "torch":
                        results.append(baseline_result)
                        continue

                    try:
                        result, output = _benchmark_backend(
                            model_name=model_name,
                            partition=partition,
                            backend=backend,
                            case_name=case_name,
                            tokens=tokens,
                            device=device,
                            dtype=dtype,
                            input_ids=input_ids,
                            positions=positions,
                            warmup=warmup,
                            repeat=repeat,
                            baseline_out=baseline_out,
                            atol=atol,
                            rtol=rtol,
                        )
                    except RuntimeError as exc:
                        if not str(exc).startswith("No available "):
                            raise
                        print(f"Skipping {model_name}/{partition}/{backend}: {exc}")
                    else:
                        results.append(result)
                        del output
                    _sync(device)

            del baseline_out
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
    finite_abs_diffs = [
        r.max_abs_diff for r in runs
        if r.max_abs_diff is not None and math.isfinite(r.max_abs_diff)
    ]
    finite_rel_diffs = [
        r.max_rel_diff for r in runs
        if r.max_rel_diff is not None and math.isfinite(r.max_rel_diff)
    ]
    return {
        "partition": runs[0].partition,
        "backend": runs[0].backend,
        "backend_status": runs[0].backend_status,
        "kernels": kernel_map,
        "aggregate_mean_ms": round(aggregate, 3),
        "speedup_vs_baseline": round(baseline_aggregate / aggregate, 4) if aggregate > 0 else None,
        "ok": all(r.ok for r in runs),
        "max_abs_diff": max(finite_abs_diffs) if finite_abs_diffs else math.inf,
        "max_rel_diff": max(finite_rel_diffs) if finite_rel_diffs else math.inf,
        "cases": {
            r.case: {
                "tokens": r.tokens,
                "mean_ms": round(r.mean_ms, 3),
                "min_ms": round(r.min_ms, 3),
                "max_ms": round(r.max_ms, 3),
                "tokens_per_second": round(r.tokens_per_second, 1),
                "max_abs_diff": r.max_abs_diff,
                "max_rel_diff": r.max_rel_diff,
                "ok": r.ok,
                "vs_baseline": round(
                    baseline_ms.get(r.case, r.mean_ms) / r.mean_ms, 4
                ) if r.mean_ms > 0 else None,
            }
            for r in runs
        },
    }


def _discover_dispatched_ops(results: list[Result], partition: str) -> list[str]:
    for r in results:
        if r.partition != partition or r.backend in ("torch", TORCH_COMPILE_BACKEND):
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
            key=lambda kv: (not all(r.ok for r in kv[1]), sum(r.mean_ms for r in kv[1])),
        )
        backend_name, runs = best_backend
        kernel_map = None
        if backend_name in CONCRETE_BACKENDS:
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
                max_abs_diff=r.max_abs_diff,
                max_rel_diff=r.max_rel_diff,
                ok=r.ok,
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
        key=lambda kv: (not all(r.ok for r in kv[1]), sum(r.mean_ms for r in kv[1])),
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
               "mean_ms", "tok/s", "diff", "rel", "ok"]
    rows = []
    for r in results:
        rows.append([
            r.model, r.partition, r.backend, r.backend_status, r.case,
            str(r.tokens), f"{r.mean_ms:.3f}", f"{r.tokens_per_second:.1f}",
            _format_diff(r.max_abs_diff),
            _format_diff(r.max_rel_diff),
            "yes" if r.ok else "no",
        ])
    widths = [
        max(len(str(row[i])) for row in [headers, *rows])
        for i in range(len(headers))
    ]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))


def _format_diff(value: float | None) -> str:
    if value is None:
        return "-"
    if not math.isfinite(value):
        return "inf"
    return f"{value:.3e}"


def _render_markdown(report: dict, slug: str) -> str:
    hw = report["hardware"]
    winner = report["winner"]
    weights_mode = report.get("weights_mode", "unknown")
    lines: list[str] = []

    title = "GSM8K Synthetic Shape Benchmark" if weights_mode == "dummy" else "GSM8K Benchmark"
    lines.append(f"# {title}: {slug}")
    lines.append("")
    lines.append(f"**GPU**: {hw.get('gpu', 'N/A')} x{hw.get('gpu_count', 1)}")
    lines.append(f"**CUDA**: {hw.get('cuda', 'N/A')}")
    lines.append(f"**PyTorch**: {hw.get('torch', 'N/A')}")
    lines.append(f"**dtype**: {report.get('dtype', 'N/A')}")
    lines.append(f"**Date**: {report.get('timestamp', 'N/A')}")
    if weights_mode == "dummy":
        lines.append("**Weights**: dummy synthetic")
        lines.append(
            "**Validation**: synthetic outputs are compared with `baseline/torch` "
            "logits for each row. This is not a GSM8K accuracy score."
        )
    else:
        lines.append("**Weights**: real")
        lines.append("**Validation**: each row is evaluated against GSM8K ground truth.")
    lines.append("")

    lines.append("## Winner")
    lines.append("")
    speedup = winner.get("speedup_vs_baseline")
    speedup_str = f" ({speedup:.3f}x vs baseline)" if speedup is not None else ""
    lines.append(f"**{winner['partition']}/{winner['backend']}**{speedup_str}")
    lines.append(f"Aggregate: {winner['aggregate_mean_ms']:.1f}ms")
    if winner.get("kernels"):
        lines.append("")
        lines.append("Kernel dispatch:")
        for op, backend in sorted(winner["kernels"].items()):
            lines.append(f"- `{op}` -> {backend}")
    lines.append("")

    lines.append("## Leaderboard")
    lines.append("")
    headers = ["#", "partition", "backend", "total (ms)", "vs baseline", "validation", "max diff"]
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
            "pass" if entry.get("ok") else "fail",
            _format_diff(entry.get("max_abs_diff")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if report["leaderboard"]:
        first_cases = report["leaderboard"][0]["cases"]
        lines.append("## Cases")
        lines.append("")
        for case_name, case_data in first_cases.items():
            validation = "pass" if case_data.get("ok") else "fail"
            lines.append(
                f"- **{case_name}**: {case_data['tokens']} tokens, "
                f"{validation}, max diff {_format_diff(case_data.get('max_abs_diff'))}"
            )
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
    parser = argparse.ArgumentParser(description="GSM8K benchmark harness")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--rtol", type=float, default=1.0e-2)
    parser.add_argument("--partition", default="all", help="all, baseline, or comma-separated hashes")
    parser.add_argument(
        "--backend",
        default="all",
        help="all, or comma-separated: torch,torch.compile,triton,cutedsl,helion",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token. Falls back to HF_TOKEN or hf_token=... in ~/.env.",
    )
    parser.add_argument(
        "--dummy-weights",
        action="store_true",
        help="Run the local synthetic-weight shape benchmark without writing a report.",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    if args.dummy_weights:
        print(
            "Running dummy-weight synthetic benchmark. No markdown report will "
            "be written and these are not GSM8K accuracy scores."
        )
    else:
        try:
            hf_token = require_hf_token(args.hf_token, purpose="GSM8K benchmark reports")
        except ValueError as exc:
            parser.error(str(exc))
        del hf_token
        raise SystemExit(REAL_WEIGHT_UNSUPPORTED)

    results = run(
        args.device,
        args.dtype,
        args.warmup,
        args.repeat,
        args.partition,
        args.backend,
        args.atol,
        args.rtol,
    )
    _print_table(results)

    models_in_results = sorted(set(r.model for r in results))
    for model_name in models_in_results:
        model_results = [r for r in results if r.model == model_name]
        report = pick_winner(model_results)
        report["weights_mode"] = "dummy"
        winner = report["winner"]
        speedup = winner.get("speedup_vs_baseline")
        speedup_str = f" ({speedup:.3f}x vs baseline)" if speedup is not None else ""
        print(f"\n[{model_name}] Winner: {winner['partition']}/{winner['backend']} "
              f"({winner['aggregate_mean_ms']:.1f}ms aggregate){speedup_str}")
        print("Report not saved in --dummy-weights mode.")


if __name__ == "__main__":
    main()
