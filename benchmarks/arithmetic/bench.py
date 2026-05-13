"""Arithmetic identity benchmark.

Uses short deterministic arithmetic prompts such as:
  1 * 213813290183291 =

The flattened racetrack models do not use real checkpoint weights or generate
text. This runner benchmarks short arithmetic-shaped token sequences for
partition/kernel latency and, when available, reports real checkpoint
correctness from benchmarks.arithmetic.eval.

Usage:
    python -m benchmarks.arithmetic.bench
    python -m benchmarks.arithmetic.bench --device cpu --kernel-filter torch
    python -m benchmarks.arithmetic.bench --kernel-filter all
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

BENCHMARK_DIR = Path(__file__).parent
MODELS = ("dsv3_2",)
CONCRETE_BACKENDS = ("triton", "cutedsl", "helion")
KERNEL_FILTERS = (
    "available",
    "all",
    "torch",
    "triton",
    "cutedsl",
    "cutedl",
    "helion",
    "best",
)

MODEL_OVERRIDES: dict[str, int | float | str] = {
    "vocab_size": 512,
    "hidden_size": 256,
    "num_attention_heads": 4,
    "head_dim": 64,
    "q_lora_rank": 64,
    "kv_lora_rank": 64,
    "qk_rope_head_dim": 16,
    "moe_intermediate_size": 128,
    "num_layers": 1,
    "n_routed_experts": 4,
    "num_experts_per_tok": 1,
    "n_shared_experts": 1,
}


@dataclass(frozen=True)
class ArithmeticCase:
    name: str
    prompt: str
    expected: str

    @property
    def text(self) -> str:
        return f"{self.prompt} {self.expected}"

    @property
    def tokens(self) -> int:
        return len(self.text.encode("ascii"))


CASES: tuple[ArithmeticCase, ...] = (
    ArithmeticCase("mul_one_long", "1 * 213813290183291 =", "213813290183291"),
    ArithmeticCase("add_zero_long", "0 + 759002341987123 =", "759002341987123"),
    ArithmeticCase("sub_zero_long", "480129381029381 - 0 =", "480129381029381"),
    ArithmeticCase("div_one_long", "98273465000123 / 1 =", "98273465000123"),
)


@dataclass
class Result:
    model: str
    partition: str
    backend: str
    backend_status: str
    case: str
    prompt: str
    expected: str
    tokens: int
    device: str
    dtype: str
    mean_ms: float
    min_ms: float
    max_ms: float
    tokens_per_second: float
    kernels: dict[str, str] | None = None


def _validate_cases(cases: tuple[ArithmeticCase, ...] = CASES) -> None:
    for case in cases:
        equation = case.prompt.removesuffix("=").strip()
        left_raw, op, right_raw = equation.split()
        left = int(left_raw)
        right = int(right_raw)
        if op == "*":
            value = left * right
        elif op == "+":
            value = left + right
        elif op == "-":
            value = left - right
        elif op == "/":
            if right == 0 or left % right != 0:
                raise ValueError(f"{case.name} is not an integer division identity")
            value = left // right
        else:
            raise ValueError(f"{case.name} uses unsupported operator {op!r}")
        if str(value) != case.expected:
            raise ValueError(
                f"{case.name} expected {case.expected!r}, but equation evaluates to {value}"
            )


def _load_partition_module(model_name: str, partition: str):
    if partition == "baseline":
        return importlib.import_module(f"partitions.{model_name}.model")
    return importlib.import_module(f"partitions.{model_name}.{partition}.model")


def _discover_partitions(model_name: str, partition_filter: str) -> list[str]:
    if partition_filter == "baseline":
        return ["baseline"]
    root = Path(__file__).resolve().parents[2] / "partitions" / model_name
    if partition_filter == "tracked":
        repo_root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            ["git", "ls-files", f"partitions/{model_name}/*/model.py"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        partitions = sorted(
            Path(line).parent.name
            for line in completed.stdout.splitlines()
            if line.strip()
        )
        return ["baseline", *partitions]
    partitions = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "model.py").exists() and not p.name.startswith("__")
    )
    if partition_filter == "all":
        return ["baseline", *partitions]
    if partition_filter not in partitions:
        raise KeyError(f"Unknown partition {partition_filter!r} for {model_name}")
    return [partition_filter]


def _backend_list(
    model_name: str,
    partition: str,
    kernel_filter: str,
    device: torch.device,
) -> list[str]:
    if partition == "baseline":
        return ["torch"]
    if kernel_filter == "cutedl":
        kernel_filter = "cutedsl"
    if kernel_filter == "all":
        return ["torch", *CONCRETE_BACKENDS]
    if kernel_filter == "available":
        backends = ["torch"]
        if device.type != "cuda":
            return backends
        module = _load_partition_module(model_name, partition)
        model = module.build_model(**MODEL_OVERRIDES).eval()
        status_map = getattr(model, "backend_status", {"torch": "native"})
        backends.extend(
            backend
            for backend in CONCRETE_BACKENDS
            if status_map.get(backend) == "native"
        )
        del model
        return backends
    if kernel_filter not in KERNEL_FILTERS:
        known = ", ".join(KERNEL_FILTERS)
        raise KeyError(f"Unknown kernel filter {kernel_filter!r}. Known: {known}")
    return [kernel_filter]


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if name not in mapping:
        raise KeyError(f"Unknown dtype {name!r}")
    return mapping[name]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _encode_case(case: ArithmeticCase, *, device: torch.device, vocab_size: int) -> torch.Tensor:
    ids = [byte % vocab_size for byte in case.text.encode("ascii")]
    return torch.tensor(ids, device=device, dtype=torch.long)


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

    times: list[float] = []
    for _ in range(repeat):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(input_ids, positions)
            end.record()
            torch.cuda.synchronize(device)
            times.append(float(start.elapsed_time(end)))
        else:
            start_time = time.perf_counter()
            output = model(input_ids, positions)
            times.append((time.perf_counter() - start_time) * 1000.0)
    assert output is not None
    return output, times


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


def _benchmark_backend(
    *,
    model_name: str,
    partition: str,
    backend: str,
    case: ArithmeticCase,
    device: torch.device,
    dtype: torch.dtype,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    warmup: int,
    repeat: int,
) -> Result:
    os.environ["RACETRACK_KERNEL_BACKEND"] = backend
    module = _load_partition_module(model_name, partition)
    model = module.build_model(**MODEL_OVERRIDES).to(device=device, dtype=dtype).eval()
    status_map = getattr(model, "backend_status", {"torch": "native"})
    backend_status = status_map.get(backend, "native")

    output, times = _time_forward(
        model,
        input_ids,
        positions,
        warmup=warmup,
        repeat=repeat,
        device=device,
    )
    dispatcher = getattr(model, "dispatcher", None)
    if backend == "best":
        best_summary = getattr(dispatcher, "best_summary", None)
        if callable(best_summary):
            backend_status = best_summary()

    kernel_map = None
    if dispatcher is not None and backend in CONCRETE_BACKENDS:
        kernel_map = _discover_kernel_map(dispatcher, backend)

    mean_ms = sum(times) / len(times)
    result = Result(
        model=model_name,
        partition=partition,
        backend=backend,
        backend_status=backend_status,
        case=case.name,
        prompt=case.prompt,
        expected=case.expected,
        tokens=case.tokens,
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        mean_ms=mean_ms,
        min_ms=min(times),
        max_ms=max(times),
        tokens_per_second=case.tokens / (mean_ms / 1000.0),
        kernels=kernel_map,
    )
    del model, output
    _sync(device)
    return result


def run(
    device_str: str = "cuda:0",
    dtype_str: str = "auto",
    warmup: int = 1,
    repeat: int = 3,
    partition_filter: str = "tracked",
    kernel_filter: str = "torch",
) -> list[Result]:
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    _validate_cases()
    torch.set_grad_enabled(False)

    device = torch.device(device_str)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")
        torch.cuda.set_device(device)
    dtype = _resolve_dtype(dtype_str, device)
    vocab_size = int(MODEL_OVERRIDES["vocab_size"])

    results: list[Result] = []
    for model_name in MODELS:
        partitions = _discover_partitions(model_name, partition_filter)
        for case in CASES:
            input_ids = _encode_case(case, device=device, vocab_size=vocab_size)
            positions = torch.arange(input_ids.numel(), device=device, dtype=torch.long)

            for partition in partitions:
                for backend in _backend_list(model_name, partition, kernel_filter, device):
                    results.append(
                        _benchmark_backend(
                            model_name=model_name,
                            partition=partition,
                            backend=backend,
                            case=case,
                            device=device,
                            dtype=dtype,
                            input_ids=input_ids,
                            positions=positions,
                            warmup=warmup,
                            repeat=repeat,
                        )
                    )
            _sync(device)
    return results


def _combo_key(r: Result) -> tuple[str, str, str]:
    return (r.model, r.partition, r.backend)


def _single_backend_mixed_plan(status: str) -> str | None:
    if not status.startswith("mixed="):
        return None
    plan = status.removeprefix("mixed=")
    if ";" in plan or "=" in plan:
        return None
    return plan if plan in CONCRETE_BACKENDS else None


def _discover_dispatched_ops(results: list[Result], partition: str) -> list[str]:
    for result in results:
        if result.partition != partition or result.backend == "torch":
            continue
        module = _load_partition_module(result.model, partition)
        model = module.build_model(**MODEL_OVERRIDES)
        dispatcher = getattr(model, "dispatcher", None)
        if dispatcher is None:
            return []
        ops: list[str] = []
        for backend in CONCRETE_BACKENDS:
            for mod in dispatcher._load_backend_modules(backend):
                if not bool(getattr(mod, "BACKEND_AVAILABLE", False)):
                    continue
                for name in sorted(dir(mod)):
                    if (
                        not name.startswith("_")
                        and name != "BACKEND_AVAILABLE"
                        and callable(getattr(mod, name))
                        and name not in ops
                    ):
                        ops.append(name)
        del model
        return ops
    return []


def _synthesize_best(results: list[Result]) -> list[Result]:
    by_partition: dict[str, dict[str, list[Result]]] = {}
    for result in results:
        by_partition.setdefault(result.partition, {}).setdefault(result.backend, []).append(result)

    best_results: list[Result] = []
    for partition, backends in by_partition.items():
        if partition == "baseline" or not any(
            backend in CONCRETE_BACKENDS for backend in backends
        ):
            continue
        concrete_backends = {backend for backend in backends if backend != "best"}
        best_backend, runs = min(
            backends.items(),
            key=lambda kv: sum(result.mean_ms for result in kv[1]),
        )
        if best_backend == "best":
            single_backend = _single_backend_mixed_plan(runs[0].backend_status)
            if single_backend in concrete_backends:
                continue
        dispatched_ops = _discover_dispatched_ops(results, partition)
        kernel_map = {op: best_backend for op in dispatched_ops} if dispatched_ops else None
        for result in runs:
            best_results.append(
                Result(
                    model=result.model,
                    partition=result.partition,
                    backend="best",
                    backend_status=f"={best_backend}",
                    case=result.case,
                    prompt=result.prompt,
                    expected=result.expected,
                    tokens=result.tokens,
                    device=result.device,
                    dtype=result.dtype,
                    mean_ms=result.mean_ms,
                    min_ms=result.min_ms,
                    max_ms=result.max_ms,
                    tokens_per_second=result.tokens_per_second,
                    kernels=kernel_map,
                )
            )
    return best_results


def _build_combo_entry(runs: list[Result], baseline_ms: dict[str, float]) -> dict:
    kernel_map = next((result.kernels for result in runs if result.kernels), None)
    aggregate = sum(result.mean_ms for result in runs)
    baseline_aggregate = sum(baseline_ms.get(result.case, result.mean_ms) for result in runs)
    return {
        "partition": runs[0].partition,
        "backend": runs[0].backend,
        "backend_status": runs[0].backend_status,
        "kernels": kernel_map,
        "aggregate_mean_ms": round(aggregate, 3),
        "speedup_vs_baseline": round(baseline_aggregate / aggregate, 4) if aggregate > 0 else None,
        "cases": {
            result.case: {
                "prompt": result.prompt,
                "expected": result.expected,
                "tokens": result.tokens,
                "mean_ms": round(result.mean_ms, 3),
                "min_ms": round(result.min_ms, 3),
                "max_ms": round(result.max_ms, 3),
                "tokens_per_second": round(result.tokens_per_second, 1),
                "vs_baseline": round(
                    baseline_ms.get(result.case, result.mean_ms) / result.mean_ms,
                    4,
                ) if result.mean_ms > 0 else None,
            }
            for result in runs
        },
    }


def pick_winner(results: list[Result]) -> dict:
    all_results = results + _synthesize_best(results)

    combos: dict[tuple[str, str, str], list[Result]] = {}
    for result in all_results:
        combos.setdefault(_combo_key(result), []).append(result)

    baseline_key = next(
        (key for key in combos if key[1] == "baseline" and key[2] == "torch"),
        None,
    )
    baseline_ms: dict[str, float] = {}
    if baseline_key is not None:
        for result in combos[baseline_key]:
            baseline_ms[result.case] = result.mean_ms

    ranked = sorted(
        combos.items(),
        key=lambda kv: sum(result.mean_ms for result in kv[1]),
    )
    _, winner_runs = ranked[0]

    return {
        "winner": _build_combo_entry(winner_runs, baseline_ms),
        "baseline": {case: round(ms, 3) for case, ms in baseline_ms.items()},
        "leaderboard": [_build_combo_entry(runs, baseline_ms) for _, runs in ranked],
        "model": winner_runs[0].model,
        "dtype": winner_runs[0].dtype,
        "hardware": _hardware_info(winner_runs[0].device),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_overrides": MODEL_OVERRIDES,
    }


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
    return f"{gpu_count}x{name.replace(' ', '_')}"


def _eval_case_map(eval_result: dict | None) -> dict[str, dict]:
    if not eval_result:
        return {}
    return {
        str(case_result.get("name")): case_result
        for case_result in eval_result.get("cases", [])
        if case_result.get("name") is not None
    }


def _format_correctness(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}%"


def _case_correctness(eval_result: dict | None, case_name: str) -> float | None:
    case_result = _eval_case_map(eval_result).get(case_name)
    if case_result is None:
        return None
    value = case_result.get("correctness_pct")
    return float(value) if value is not None else None


def _load_cached_eval() -> dict | None:
    path = BENCHMARK_DIR / "results" / "eval_cache.json"
    if not path.exists():
        return None
    try:
        import json as _json

        cache = _json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(cache, dict) or not cache:
        return None
    latest_key = sorted(cache)[-1]
    result = cache.get(latest_key)
    return result if isinstance(result, dict) else None


def _print_table(results: list[Result], eval_result: dict | None = None) -> None:
    headers = [
        "model",
        "partition",
        "backend",
        "status",
        "case",
        "tokens",
        "mean_ms",
        "tok/s",
        "correctness",
    ]
    rows = [
        [
            result.model,
            result.partition,
            result.backend,
            result.backend_status,
            result.case,
            str(result.tokens),
            f"{result.mean_ms:.3f}",
            f"{result.tokens_per_second:.1f}",
            _format_correctness(_case_correctness(eval_result, result.case)),
        ]
        for result in results
    ]
    widths = [
        max(len(str(row[i])) for row in [headers, *rows])
        for i in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))


def _render_markdown(report: dict, slug: str) -> str:
    hw = report["hardware"]
    winner = report["winner"]
    eval_result = report.get("eval")
    lines: list[str] = []

    lines.append(f"# Arithmetic Benchmark: {slug}")
    lines.append("")
    lines.append(f"**GPU**: {hw.get('gpu', 'N/A')} x{hw.get('gpu_count', 1)}")
    lines.append(f"**CUDA**: {hw.get('cuda', 'N/A')}")
    lines.append(f"**PyTorch**: {hw.get('torch', 'N/A')}")
    lines.append(f"**dtype**: {report.get('dtype', 'N/A')}")
    lines.append(f"**Date**: {report.get('timestamp', 'N/A')}")
    if eval_result:
        lines.append(f"**Eval model**: {eval_result.get('model', 'N/A')}")
        lines.append(
            f"**Arithmetic accuracy**: "
            f"{_format_correctness(float(eval_result.get('accuracy_pct', 0.0)))} "
            f"({eval_result.get('correct', 0)}/{eval_result.get('num_samples', 0)})"
        )
    else:
        lines.append("**Arithmetic accuracy**: not run")
    lines.append("")

    speedup = winner.get("speedup_vs_baseline")
    speedup_str = f" ({speedup:.3f}x vs baseline)" if speedup is not None else ""
    lines.append("## Winner")
    lines.append("")
    lines.append(f"**{winner['partition']}/{winner['backend']}**{speedup_str}")
    lines.append(f"Aggregate: {winner['aggregate_mean_ms']:.3f}ms")
    if winner.get("kernels"):
        lines.append("")
        lines.append("Kernel dispatch:")
        for op, backend in sorted(winner["kernels"].items()):
            lines.append(f"- `{op}` -> {backend}")
    lines.append("")

    lines.append("## Leaderboard")
    lines.append("")
    headers = ["#", "partition", "backend", "total (ms)", "vs baseline", "correctness"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for rank, entry in enumerate(report["leaderboard"], 1):
        speedup_val = entry.get("speedup_vs_baseline")
        speedup_cell = f"{speedup_val:.3f}x" if speedup_val is not None else "-"
        kernels_note = ""
        if entry["backend"] == "best" and entry.get("kernels"):
            kernels_note = " (" + ", ".join(
                f"{op}={backend}" for op, backend in sorted(entry["kernels"].items())
            ) + ")"
        row = [
            str(rank),
            entry["partition"],
            f"{entry['backend']}{kernels_note}",
            f"{entry['aggregate_mean_ms']:.3f}",
            speedup_cell,
            _format_correctness(
                float(eval_result["accuracy_pct"]) if eval_result else None
            ),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Cases")
    lines.append("")
    lines.append("| case | prompt | expected | tokens | correctness |")
    lines.append("|---|---|---|---|---|")
    first_cases = report["leaderboard"][0]["cases"] if report["leaderboard"] else {}
    for case_name, case_data in first_cases.items():
        lines.append(
            f"| {case_name} | `{case_data['prompt']}` | `{case_data['expected']}` | "
            f"{case_data['tokens']} | "
            f"{_format_correctness(_case_correctness(eval_result, case_name))} |"
        )
    lines.append("")

    baseline = report.get("baseline", {})
    if baseline:
        lines.append("## Baseline Reference")
        lines.append("")
        for case_name, ms in baseline.items():
            lines.append(f"- {case_name}: {ms:.3f}ms")
        lines.append("")

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arithmetic identity benchmark")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--partition",
        default="tracked",
        help="baseline, tracked, all, or a partition hash",
    )
    parser.add_argument(
        "--kernel-filter",
        default="torch",
        choices=KERNEL_FILTERS,
        help="available, all, torch, triton, cutedsl/cutedl, helion, or best",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--no-eval", action="store_true", help="Do not load cached real-weight eval")
    parser.add_argument("--no-save", action="store_true", help="Do not write results/<hardware>.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = run(
        device_str=args.device,
        dtype_str=args.dtype,
        warmup=args.warmup,
        repeat=args.repeat,
        partition_filter=args.partition,
        kernel_filter=args.kernel_filter,
    )
    eval_result = None
    if not args.no_eval:
        eval_result = _load_cached_eval()
        if eval_result is not None:
            print(
                f"Loaded cached arithmetic eval: "
                f"{eval_result['accuracy_pct']}% "
                f"({eval_result['correct']}/{eval_result['num_samples']})"
            )
        else:
            print(
                "No cached arithmetic eval results. Run:\n"
                "  torchrun --standalone --nproc-per-node=8 \\\n"
                "    -m benchmarks.arithmetic.eval \\\n"
                "    --ckpt-path checkpoints/dsv3_2-mp8"
            )
    _print_table(results, eval_result)

    report = pick_winner(results)
    if eval_result is not None:
        report["eval"] = eval_result
    winner = report["winner"]
    slug = _hardware_slug(args.device)
    if not args.no_save:
        results_dir = BENCHMARK_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        results_path = results_dir / f"{slug}.md"
        results_path.write_text(_render_markdown(report, slug))
        print(f"Saved to {results_path}")
    speedup = winner.get("speedup_vs_baseline")
    speedup_str = f" ({speedup:.3f}x vs baseline)" if speedup is not None else ""
    print(
        f"\nWinner: {winner['partition']}/{winner['backend']} "
        f"({winner['aggregate_mean_ms']:.3f}ms aggregate){speedup_str}"
    )


if __name__ == "__main__":
    main()
