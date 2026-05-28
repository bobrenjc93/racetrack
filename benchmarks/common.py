"""Shared benchmark utilities used by arithmetic and GSM8K benchmarks.

Consolidates duplicated code: backend normalization, hardware info,
timing, partition discovery, result ranking, and report rendering.
"""
from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TORCH_COMPILE_BACKEND = "torch.compile"
TORCH_COMPILE_ALIASES = {TORCH_COMPILE_BACKEND, "torch_compile"}
CONCRETE_BACKENDS = ("triton", "cutedsl", "helion")
MODELS = ("dsv3_2", "dsv3_2_nvfp4")

EVAL_MODEL = "deepseek-ai/DeepSeek-V3.2"

DSV3_2_CONFIG = {
    "vocab_size": 129280,
    "dim": 7168,
    "inter_dim": 18432,
    "moe_inter_dim": 2048,
    "n_layers": 61,
    "n_dense_layers": 3,
    "n_heads": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "n_activated_experts": 8,
    "n_expert_groups": 8,
    "n_limited_groups": 4,
    "score_func": "sigmoid",
    "route_scale": 2.5,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "original_seq_len": 4096,
    "rope_theta": 10000.0,
    "rope_factor": 40,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 2048,
}


def normalize_backend_name(backend: str) -> str:
    backend = backend.strip().lower()
    if backend in TORCH_COMPILE_ALIASES:
        return TORCH_COMPILE_BACKEND
    if backend == "cutedl":
        return "cutedsl"
    return backend


def env_backend(backend: str) -> str:
    return "torch" if backend == TORCH_COMPILE_BACKEND else backend


def compile_model_if_requested(model: nn.Module, backend: str) -> nn.Module:
    if backend != TORCH_COMPILE_BACKEND:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    return torch.compile(model)


def cleanup_compile_state(device: torch.device | None = None) -> None:
    dynamo = getattr(torch, "_dynamo", None)
    reset = getattr(dynamo, "reset", None)
    if callable(reset):
        reset()
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
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


def time_forward(
    model: nn.Module,
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
    sync(device)

    if device.type == "cuda":
        if output is None:
            output = model(input_ids, positions)
            sync(device)
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


def hardware_info(device_str: str = "cuda:0") -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": device_str,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        dev = torch.device(device_str)
        props = torch.cuda.get_device_properties(dev)
        info["gpu"] = props.name
        info["gpu_count"] = torch.cuda.device_count()
        total_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        info["gpu_memory_gb"] = round(total_mem / 1e9, 1)
        cuda_version = torch.version.cuda
        if cuda_version:
            info["cuda"] = cuda_version
    return info


def hardware_slug(device_str: str = "cuda:0") -> str:
    if not torch.cuda.is_available():
        return "cpu"
    props = torch.cuda.get_device_properties(torch.device(device_str))
    name = props.name.lower()
    for prefix in ("nvidia ", "amd ", "intel "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.replace(" ", "_")
    count = torch.cuda.device_count()
    return f"{count}x{name}"


def discover_partition_hashes(
    model_name: str,
    partition_filter: str = "all",
) -> list[str]:
    root = PROJECT_ROOT / "partitions" / model_name
    if not root.is_dir():
        return []
    partitions = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir()
        and (p / "kernels").is_dir()
        and ((p / "spec.py").exists() or (p / "model.py").exists())
        and not p.name.startswith("__")
    )
    if partition_filter == "all":
        return partitions
    if partition_filter in partitions:
        return [partition_filter]
    wanted = {h.strip() for h in partition_filter.split(",") if h.strip()}
    matched = [p for p in partitions if p in wanted]
    if not matched:
        raise KeyError(f"No partitions matched filter {partition_filter!r}")
    return matched


def discover_kernel_map(dispatcher: Any, backend: str) -> dict[str, str] | None:
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


def format_diff(value: float | None) -> str:
    if value is None:
        return "-"
    import math
    if not math.isfinite(value):
        return "inf"
    return f"{value:.3e}"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        print(fmt.format(*padded))
