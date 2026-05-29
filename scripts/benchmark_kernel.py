#!/usr/bin/env python3
"""Per-kernel micro-benchmark with backend leaderboard.

Runs a single fused op across all available backends (triton, helion,
cutedsl) with realistic tensor shapes, showing timing, memory bandwidth,
and compute efficiency.

Usage:
    python scripts/benchmark_kernel.py --partition 3336cdbd --kernel fused_act_quant
    python scripts/benchmark_kernel.py --partition 3336cdbd --kernel fused_residual_norm --backend triton
    python scripts/benchmark_kernel.py --partition 3336cdbd --kernel fused_act_quant --shape 1,1,7168
    python scripts/benchmark_kernel.py --list-kernels --partition 3336cdbd
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from racetrack.partition_spec import load_spec
from racetrack.runtime.dispatch import KernelDispatcher

BACKENDS = ("triton", "cutedsl", "helion")

# Realistic shapes for each kernel during decode (batch=1, world_size=8)
DEFAULT_SHAPES: dict[str, dict] = {
    "fused_act_quant": {
        "shapes": {
            "model_dim": {"x": (1, 1, 2048)},
            "moe_expert": {"x": (1, 2048)},
            "mla_qkv": {"x": (1, 1, 7168)},
        },
        "dtype": torch.bfloat16,
        "gen": lambda shape, dtype: {"x": torch.randn(shape["x"], dtype=dtype, device="cuda")},
        "call": lambda fn, inputs: fn(inputs["x"], fallback=None),
        "bytes": lambda inputs, outputs: (
            inputs["x"].numel() * inputs["x"].element_size()
            + outputs[0].numel() * 1
            + outputs[1].numel() * 4
        ),
    },
    "fused_swiglu_quant": {
        "shapes": {
            "dense_mlp": {"gate": (1, 1, 1368), "up": (1, 1, 1368)},
            "moe_expert": {"gate": (1, 1408), "up": (1, 1408)},
        },
        "dtype": torch.bfloat16,
        "gen": lambda shape, dtype: {
            "gate": torch.randn(shape["gate"], dtype=dtype, device="cuda"),
            "up": torch.randn(shape["up"], dtype=dtype, device="cuda"),
        },
        "call": lambda fn, inputs: fn(inputs["gate"], inputs["up"], fallback=None),
        "bytes": lambda inputs, outputs: (
            inputs["gate"].numel() * inputs["gate"].element_size()
            + inputs["up"].numel() * inputs["up"].element_size()
            + outputs[0].numel() * 1
            + outputs[1].numel() * 4
        ),
    },
    "fused_residual_norm": {
        "shapes": {
            "model_dim": {"update": (1, 1, 2048), "residual": (1, 1, 2048), "weight": (2048,)},
            "large_dim": {"update": (1, 1, 7168), "residual": (1, 1, 7168), "weight": (7168,)},
        },
        "dtype": torch.bfloat16,
        "gen": lambda shape, dtype: {
            "update": torch.randn(shape["update"], dtype=dtype, device="cuda"),
            "residual": torch.randn(shape["residual"], dtype=dtype, device="cuda"),
            "weight": torch.ones(shape["weight"], dtype=torch.float32, device="cuda"),
        },
        "call": lambda fn, inputs: fn(
            inputs["update"], inputs["residual"], inputs["weight"],
            eps=1e-6, fallback=None,
        ),
        "bytes": lambda inputs, outputs: (
            inputs["update"].numel() * inputs["update"].element_size()
            + inputs["residual"].numel() * inputs["residual"].element_size()
            + inputs["weight"].numel() * 4
            + outputs[0].numel() * outputs[0].element_size()
            + outputs[1].numel() * outputs[1].element_size()
        ),
    },
    "fused_swiglu": {
        "shapes": {
            "dense_mlp": {"gate": (1, 1, 1368), "up": (1, 1, 1368)},
            "moe_expert": {"gate": (1, 1408), "up": (1, 1408)},
        },
        "dtype": torch.bfloat16,
        "gen": lambda shape, dtype: {
            "gate": torch.randn(shape["gate"], dtype=dtype, device="cuda"),
            "up": torch.randn(shape["up"], dtype=dtype, device="cuda"),
        },
        "call": lambda fn, inputs: fn(inputs["gate"], inputs["up"], fallback=None),
        "bytes": lambda inputs, outputs: (
            inputs["gate"].numel() * inputs["gate"].element_size()
            + inputs["up"].numel() * inputs["up"].element_size()
            + outputs.numel() * outputs.element_size()
        ),
    },
}

H100_HBM_BW_GB_S = 3350.0
H100_PEAK_TFLOPS_BF16 = 989.0


def _time_kernel(fn, inputs, call_fn, *, warmup=50, repeat=200):
    """Time a kernel with CUDA events. Returns median time in microseconds."""
    for _ in range(warmup):
        call_fn(fn, inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = call_fn(fn, inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms → µs

    times.sort()
    n = len(times)
    return {
        "median_us": times[n // 2],
        "min_us": times[0],
        "p25_us": times[n // 4],
        "p75_us": times[3 * n // 4],
        "max_us": times[-1],
        "result": result,
    }


def _format_row(row: dict) -> list[str]:
    return [
        row["backend"],
        row["shape_name"],
        f"{row['median_us']:.2f}",
        f"{row['min_us']:.2f}",
        f"{row['p75_us']:.2f}",
        f"{row['bw_gb_s']:.1f}",
        f"{row['bw_pct']:.0f}%",
        f"{row['speedup']:.2f}x",
    ]


def benchmark_kernel(
    model: str,
    partition: str,
    kernel: str,
    backends: list[str],
    custom_shape: tuple[int, ...] | None = None,
    warmup: int = 50,
    repeat: int = 200,
):
    spec = load_spec(model, partition)
    dispatcher = KernelDispatcher(spec.kernel_root)

    if kernel not in DEFAULT_SHAPES:
        print(f"Unknown kernel: {kernel}")
        print(f"Available: {', '.join(sorted(DEFAULT_SHAPES.keys()))}")
        return

    kinfo = DEFAULT_SHAPES[kernel]
    dtype = kinfo["dtype"]

    if custom_shape:
        # Multi-input kernels (swiglu, residual_norm) have gen lambdas that
        # index every key of the shape spec, so a custom shape must be
        # broadcast to all keys or gen() raises KeyError before timing.
        template = list(kinfo["shapes"].values())[0]
        shapes = {"custom": {k: custom_shape for k in template}}
    else:
        shapes = kinfo["shapes"]

    results = []

    for backend in backends:
        fn = dispatcher._resolve(backend, kernel)
        if fn is None:
            continue

        for shape_name, shape_spec in shapes.items():
            inputs = kinfo["gen"](shape_spec, dtype)

            try:
                timing = _time_kernel(fn, inputs, kinfo["call"], warmup=warmup, repeat=repeat)
            except Exception as e:
                print(f"  {backend}/{shape_name}: FAILED ({e})")
                continue

            total_bytes = kinfo["bytes"](inputs, timing["result"])
            bw_gb_s = total_bytes / (timing["median_us"] / 1e6) / 1e9

            results.append({
                "backend": backend,
                "shape_name": shape_name,
                "median_us": timing["median_us"],
                "min_us": timing["min_us"],
                "p25_us": timing["p25_us"],
                "p75_us": timing["p75_us"],
                "max_us": timing["max_us"],
                "total_bytes": total_bytes,
                "bw_gb_s": bw_gb_s,
                "bw_pct": bw_gb_s / H100_HBM_BW_GB_S * 100,
                "speedup": 1.0,
            })

    if not results:
        print(f"No backends found for {kernel} in {model}/{partition}")
        return

    results.sort(key=lambda r: (r["shape_name"], r["median_us"]))

    for shape_name in sorted(set(r["shape_name"] for r in results)):
        group = [r for r in results if r["shape_name"] == shape_name]
        fastest = min(r["median_us"] for r in group)
        for r in group:
            r["speedup"] = fastest / r["median_us"]

    print(f"\n{'='*80}")
    print(f"  Kernel: {kernel}")
    print(f"  Partition: {model}/{partition}")
    print(f"  Warmup: {warmup} | Repeat: {repeat}")
    print(f"{'='*80}\n")

    headers = ["backend", "shape", "median(µs)", "min(µs)", "p75(µs)", "BW(GB/s)", "BW(%peak)", "vs best"]
    widths = [max(len(h), max((len(c) for c in col), default=0))
              for h, col in zip(headers, zip(*[_format_row(r) for r in results]))]
    widths = [max(w, len(h)) for w, h in zip(widths, headers)]

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))

    prev_shape = None
    for r in results:
        if prev_shape and r["shape_name"] != prev_shape:
            print()
        prev_shape = r["shape_name"]
        row = _format_row(r)
        line = fmt.format(*row)
        if r["speedup"] >= 0.99:
            line = f"\033[92m{line}\033[0m"
        print(line)

    print()
    for shape_name in sorted(set(r["shape_name"] for r in results)):
        group = [r for r in results if r["shape_name"] == shape_name]
        winner = min(group, key=lambda r: r["median_us"])
        shape_spec = shapes[shape_name]
        shape_str = " | ".join(f"{k}={v}" for k, v in shape_spec.items())
        print(f"  Winner ({shape_name}): {winner['backend']} @ {winner['median_us']:.2f}µs")
        print(f"    Shape: {shape_str}")
        print(f"    Bandwidth: {winner['bw_gb_s']:.1f} GB/s ({winner['bw_pct']:.0f}% of H100 peak)")
    print()


def list_kernels(model: str, partition: str):
    spec = load_spec(model, partition)
    dispatcher = KernelDispatcher(spec.kernel_root)

    print(f"\nKernels in {model}/{partition}:\n")
    print(f"  {'op':<30} {'triton':<10} {'helion':<10} {'cutedsl':<10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    all_ops = set()
    for backend in BACKENDS:
        for mod in dispatcher._load_backend_modules(backend):
            if not getattr(mod, "BACKEND_AVAILABLE", False):
                continue
            for name in dir(mod):
                if not name.startswith("_") and name != "BACKEND_AVAILABLE" and callable(getattr(mod, name)):
                    all_ops.add(name)

    for op in sorted(all_ops):
        avail = []
        for backend in BACKENDS:
            fn = dispatcher._resolve(backend, op)
            avail.append("✓" if fn else "-")
        print(f"  {op:<30} {avail[0]:<10} {avail[1]:<10} {avail[2]:<10}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Per-kernel micro-benchmark with backend leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/benchmark_kernel.py --partition 3336cdbd --kernel fused_act_quant
  python scripts/benchmark_kernel.py --partition 3336cdbd --kernel fused_residual_norm --backend triton
  python scripts/benchmark_kernel.py --partition 3336cdbd --kernel fused_act_quant --shape 1,1,7168
  python scripts/benchmark_kernel.py --list-kernels --partition 3336cdbd
  python scripts/benchmark_kernel.py --list-kernels --model dsv3_2_nvfp4 --partition f1bdaa6e
""",
    )
    parser.add_argument("--model", default="dsv3_2", help="Model name (default: dsv3_2)")
    parser.add_argument("--partition", required=True, help="Partition hash")
    parser.add_argument("--kernel", help="Kernel op name (e.g., fused_act_quant)")
    parser.add_argument("--backend", help="Specific backend (default: all available)")
    parser.add_argument("--shape", help="Custom shape as comma-separated ints (e.g., 1,1,7168)")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations (default: 50)")
    parser.add_argument("--repeat", type=int, default=200, help="Timed iterations (default: 200)")
    parser.add_argument("--list-kernels", action="store_true", help="List available kernels and exit")
    args = parser.parse_args()

    if args.list_kernels:
        list_kernels(args.model, args.partition)
        return

    if not args.kernel:
        parser.error("--kernel is required (or use --list-kernels)")

    if args.repeat < 1:
        parser.error("--repeat must be at least 1 (need a timed iteration)")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    backends = [args.backend] if args.backend else list(BACKENDS)
    custom_shape = tuple(int(x) for x in args.shape.split(",")) if args.shape else None

    benchmark_kernel(
        model=args.model,
        partition=args.partition,
        kernel=args.kernel,
        backends=backends,
        custom_shape=custom_shape,
        warmup=args.warmup,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    main()
