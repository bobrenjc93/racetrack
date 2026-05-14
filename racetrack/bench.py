from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from benchmarks import get_cases


CONCRETE_KERNEL_BACKENDS = ("triton", "cutedsl", "helion")
BACKENDS = ("torch", "triton", "cutedsl", "helion", "best", "all")
MODELS = ("dsv3_2", "dsv3_2_nvfp4")


@dataclass
class BenchResult:
    model: str
    partition: str
    backend: str
    backend_status: str
    case: str
    device: str
    tokens: int
    dtype: str
    mean_ms: float
    min_ms: float
    max_ms: float
    tokens_per_second: float
    max_abs_diff: float | None
    ok: bool


def _dtype(name: str, device: torch.device) -> torch.dtype:
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


def _device_list(args: argparse.Namespace) -> list[torch.device]:
    if args.devices:
        devices = []
        for raw in args.devices.split(","):
            raw = raw.strip()
            if raw.startswith("cuda"):
                devices.append(torch.device(raw))
            else:
                devices.append(torch.device(f"cuda:{raw}"))
        return devices
    return [torch.device(args.device)]


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
            start_time = time.perf_counter()
            output = model(input_ids, positions)
            times.append((time.perf_counter() - start_time) * 1000.0)
    assert output is not None
    return output, times


def _load_model(model_name: str, partition: str):
    if partition == "baseline":
        return importlib.import_module(f"partitions.{model_name}.model")
    return importlib.import_module(f"partitions.{model_name}.{partition}.model")


def _discover_partitions(model_name: str, partition_filter: str) -> list[str]:
    if partition_filter == "baseline":
        return ["baseline"]
    root = Path(__file__).resolve().parents[1] / "partitions" / model_name
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


def _backend_list(kernel_filter: str, partition: str) -> list[str]:
    if partition == "baseline":
        return ["torch"]
    if kernel_filter == "all":
        return list(CONCRETE_KERNEL_BACKENDS)
    if kernel_filter == "cutedl":
        return ["cutedsl"]
    if kernel_filter not in BACKENDS:
        known = ", ".join((*BACKENDS, "all", "cutedl"))
        raise KeyError(f"Unknown kernel filter {kernel_filter!r}. Known: {known}")
    return [kernel_filter]


def _model_list(model_filter: str) -> Iterable[str]:
    if model_filter not in MODELS:
        known = ", ".join(MODELS)
        raise KeyError(f"Unknown model {model_filter!r}. Known: {known}")
    return [model_filter]


def _benchmark_backend(
    *,
    model_name: str,
    partition: str,
    backend: str,
    case,
    device: torch.device,
    dtype: torch.dtype,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    baseline_out: torch.Tensor,
    warmup: int,
    repeat: int,
    check: bool,
    atol: float,
) -> BenchResult:
    os.environ["RACETRACK_KERNEL_BACKEND"] = backend
    module = _load_model(model_name, partition)
    model = module.build_model().to(device=device, dtype=dtype).eval()
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
    if backend == "best":
        dispatcher = getattr(model, "dispatcher", None)
        best_summary = getattr(dispatcher, "best_summary", None)
        if callable(best_summary):
            backend_status = best_summary()
    diff = None
    ok = True
    if check:
        diff = float((output.float() - baseline_out.float()).abs().max())
        ok = diff <= atol
    mean_ms = sum(times) / len(times)
    result = BenchResult(
        model=model_name,
        partition=partition,
        backend=backend,
        backend_status=backend_status,
        case=case.name,
        device=str(device),
        tokens=case.tokens,
        dtype=str(dtype).replace("torch.", ""),
        mean_ms=mean_ms,
        min_ms=min(times),
        max_ms=max(times),
        tokens_per_second=case.tokens / (mean_ms / 1000.0),
        max_abs_diff=diff,
        ok=ok,
    )
    del model, output
    _sync(device)
    return result


def _best_result_from_candidates(candidates: list[BenchResult]) -> BenchResult:
    concrete_backends = {result.backend for result in candidates if result.backend != "best"}
    normalized_candidates = []
    for result in candidates:
        if result.backend == "best":
            single_backend = _single_backend_mixed_plan(result.backend_status)
            if single_backend in concrete_backends:
                continue
        normalized_candidates.append(result)
    normalized_ok = [result for result in normalized_candidates if result.ok]
    best = min(normalized_ok or normalized_candidates, key=lambda result: result.mean_ms)
    backend_status = (
        best.backend_status if best.backend == "best" else f"pure={best.backend}"
    )
    return BenchResult(
        model=best.model,
        partition=best.partition,
        backend="best",
        backend_status=backend_status,
        case=best.case,
        device=best.device,
        tokens=best.tokens,
        dtype=best.dtype,
        mean_ms=best.mean_ms,
        min_ms=best.min_ms,
        max_ms=best.max_ms,
        tokens_per_second=best.tokens_per_second,
        max_abs_diff=best.max_abs_diff,
        ok=best.ok,
    )


def _single_backend_mixed_plan(status: str) -> str | None:
    if not status.startswith("mixed="):
        return None
    plan = status.removeprefix("mixed=")
    if ";" in plan or "=" in plan:
        return None
    return plan if plan in CONCRETE_KERNEL_BACKENDS else None


def _print_table(results: list[BenchResult]) -> None:
    if not results:
        return
    headers = [
        "model",
        "partition",
        "backend",
        "status",
        "case",
        "device",
        "mean_ms",
        "tok/s",
        "diff",
        "ok",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result.model,
                result.partition,
                result.backend,
                result.backend_status,
                result.case,
                result.device,
                f"{result.mean_ms:.3f}",
                f"{result.tokens_per_second:.1f}",
                "-" if result.max_abs_diff is None else f"{result.max_abs_diff:.3e}",
                "yes" if result.ok else "no",
            ]
        )
    widths = [
        max(len(str(row[i])) for row in ([headers] + rows))
        for i in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))


def run(args: argparse.Namespace) -> list[BenchResult]:
    torch.set_grad_enabled(False)
    results: list[BenchResult] = []
    cases = get_cases(args.benchmark)
    devices = _device_list(args)

    for device in devices:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = _dtype(args.dtype, device)
        for model_name in _model_list(args.model):
            for case in cases:
                repeat = args.repeat if args.repeat is not None else case.repeat
                warmup = args.warmup if args.warmup is not None else case.warmup
                input_ids = (
                    torch.arange(case.tokens, device=device, dtype=torch.long)
                    % 4096
                )
                positions = torch.arange(case.tokens, device=device, dtype=torch.long)

                baseline_module = _load_model(model_name, "baseline")
                baseline = baseline_module.build_model().to(device=device, dtype=dtype).eval()
                baseline_out, _ = _time_forward(
                    baseline,
                    input_ids,
                    positions,
                    warmup=max(1, min(warmup, 2)),
                    repeat=1,
                    device=device,
                )
                del baseline
                _sync(device)

                for partition in _discover_partitions(model_name, args.partition):
                    backends = _backend_list(args.kernel_filter, partition)
                    if args.kernel_filter == "all" and partition != "baseline":
                        candidate_results = []
                        for backend in backends:
                            result = _benchmark_backend(
                                model_name=model_name,
                                partition=partition,
                                backend=backend,
                                case=case,
                                device=device,
                                dtype=dtype,
                                input_ids=input_ids,
                                positions=positions,
                                baseline_out=baseline_out,
                                warmup=warmup,
                                repeat=repeat,
                                check=args.check,
                                atol=args.atol,
                            )
                            results.append(result)
                            candidate_results.append(result)
                        candidate_results.append(
                            _benchmark_backend(
                                model_name=model_name,
                                partition=partition,
                                backend="best",
                                case=case,
                                device=device,
                                dtype=dtype,
                                input_ids=input_ids,
                                positions=positions,
                                baseline_out=baseline_out,
                                warmup=warmup,
                                repeat=repeat,
                                check=args.check,
                                atol=args.atol,
                            )
                        )
                        results.append(_best_result_from_candidates(candidate_results))
                    else:
                        for backend in backends:
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
                                    baseline_out=baseline_out,
                                    warmup=warmup,
                                    repeat=repeat,
                                    check=args.check,
                                    atol=args.atol,
                                )
                            )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep racetrack model partitions and kernels.")
    parser.add_argument("--model", default="dsv3_2", choices=MODELS)
    parser.add_argument("--partition", default="all", help="baseline, all, or a partition hash")
    parser.add_argument(
        "--kernel-filter",
        default="all",
        help="torch, triton, cutedsl/cutedl, helion, best, or all",
    )
    parser.add_argument("--benchmark", default="smoke", help="smoke, decode_128, prefill_512, prefill_2048, or all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", default=None, help="Comma separated CUDA ordinals, e.g. 0,1,2,3")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--json", type=Path, default=None, help="Optional path for JSON results.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = run(args)
    _print_table(results)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(result) for result in results], indent=2))
    failed = [result for result in results if not result.ok]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
