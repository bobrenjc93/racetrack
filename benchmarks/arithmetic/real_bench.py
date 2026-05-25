"""Real-weight arithmetic benchmark with flat_decode + CUDA graph.

Loads the real DeepSeek-V3.2 checkpoint, runs short arithmetic prompts
through flat_decode with CUDA graph capture, and reports per-backend
decode latency.

Usage:
    torchrun --standalone --nproc-per-node=8 -m benchmarks.arithmetic.real_bench \
        --hf-direct --partition-model dsv3_2 --backend all
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.gsm8k.eval import DSV3_2_CONFIG, EVAL_MODEL
from benchmarks.gsm8k.hf_auth import require_hf_token
from benchmarks.common import (
    TORCH_COMPILE_BACKEND,
    CONCRETE_BACKENDS,
    hardware_info as _hardware_info,
    hardware_slug as _hardware_slug,
)
from benchmarks.gsm8k.real_kernels import (
    discover_real_kernel_rows,
    patch_real_model,
)


PROMPTS = [
    "1 * 213813290183291 =",
    "0 + 759002341987123 =",
    "480129381029381 - 0 =",
    "98273465000123 / 1 =",
]


def _rank():
    return int(os.environ.get("RANK", "0"))


def _world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_rank0():
    return _rank() == 0


def _load_model_and_tokenizer(*, ckpt_path, hf_token, hf_direct):
    import torch.distributed as dist
    from transformers import PreTrainedTokenizerFast
    from inference.model import ModelArgs, Transformer
    from benchmarks.gsm8k.hf_model_loader import (
        load_hf_sharded_weights,
        run_post_load_transforms,
    )

    world_size = _world_size()
    rank = _rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)

    config = dict(DSV3_2_CONFIG)
    config["max_batch_size"] = 1
    config["max_seq_len"] = 512
    config["dtype"] = "fp8"
    args = ModelArgs(**config)

    ckpt_file = ckpt_path / f"model{rank}-mp{world_size}.safetensors"
    use_converted = ckpt_file.exists()
    if not use_converted and not hf_direct:
        raise FileNotFoundError("Pass --hf-direct to load HF shards")

    if _is_rank0():
        print(f"Loading real model (world_size={world_size}) ...", flush=True)

    with torch.device("cuda"):
        model = Transformer(args)
    if use_converted:
        from safetensors.torch import load_model
        load_model(model, str(ckpt_file), strict=False)
    else:
        loaded = load_hf_sharded_weights(
            model, repo_id=EVAL_MODEL, hf_token=hf_token,
            rank=rank, world_size=world_size,
        )
        if _is_rank0():
            print(f"Loaded {loaded} HF tensors for rank 0", flush=True)
    transforms = run_post_load_transforms(model)
    if _is_rank0():
        print(f"Ran {len(transforms)} post-load model transforms", flush=True)
    model.eval()

    from huggingface_hub import hf_hub_download
    tok_path = hf_hub_download(EVAL_MODEL, "tokenizer.json", token=hf_token)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)
    return model, tokenizer


@torch.inference_mode()
def _time_baseline(model, tokenizer, prompts, *, max_new_tokens, warmup, repeat):
    from benchmarks.gsm8k.eval import _generate_greedy

    encoded = [tokenizer.encode(p) for p in prompts]
    for _ in range(warmup):
        for p in encoded:
            _generate_greedy(model, [p], max_new_tokens, eos_id=1)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for p in encoded:
            _generate_greedy(model, [p], max_new_tokens, eos_id=1)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return sum(times) / len(times)


@torch.inference_mode()
def _time_cudagraph_backends(model, tokenizer, prompts, kr, backends, *, max_new_tokens, warmup, repeat):
    from benchmarks.gsm8k.flat_decode import build_flat_decode

    encoded = [tokenizer.encode(p) for p in prompts]
    max_prompt = max(len(e) for e in encoded)
    max_seq = max_prompt + max_new_tokens

    first_prompt = encoded[0]
    tok = torch.tensor([first_prompt], dtype=torch.long, device="cuda")
    model.forward(tok, 0)
    torch.cuda.synchronize()
    prompt_len = len(first_prompt)
    n_decode_tokens = max_new_tokens * len(prompts)

    results = {}
    for backend in backends:
        try:
            flat_fn, flat_cg_fn, update_bufs, s_logits = build_flat_decode(
                model, kr, backend=backend, max_seq_len=max_seq,
            )

            static_tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
            for i in range(min(5, prompt_len)):
                update_bufs(prompt_len + i)
                static_tok.fill_(0)
                flat_cg_fn(static_tok)
            torch.cuda.synchronize()

            update_bufs(prompt_len + 10)
            flat_cg_fn(static_tok)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                flat_cg_fn(static_tok)
            torch.cuda.synchronize()

            for _ in range(warmup):
                for i in range(n_decode_tokens):
                    update_bufs(prompt_len + (i % max_new_tokens))
                    graph.replay()
            torch.cuda.synchronize()

            times = []
            for _ in range(repeat):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for i in range(n_decode_tokens):
                    update_bufs(prompt_len + (i % max_new_tokens))
                    graph.replay()
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            results[backend] = sum(times) / len(times)
        except Exception as exc:
            rank = int(os.environ.get("RANK", "0"))
            if rank == 0:
                print(f"  {backend} failed: {exc}", flush=True)
    return results


def run(*, ckpt_path, hf_token, hf_direct, partition_model, backend_filter, warmup, repeat):
    model, tokenizer = _load_model_and_tokenizer(
        ckpt_path=ckpt_path, hf_token=hf_token, hf_direct=hf_direct,
    )

    if _is_rank0():
        print("Timing baseline ...", flush=True)
    baseline_ms = _time_baseline(model, tokenizer, PROMPTS,
                                  max_new_tokens=64, warmup=2, repeat=3)

    if _is_rank0():
        print("Timing torch.compile ...", flush=True)
    compiled = torch.compile(model)
    compile_ms = _time_baseline(compiled, tokenizer, PROMPTS,
                                 max_new_tokens=64, warmup=2, repeat=3)
    del compiled
    from benchmarks.common import cleanup_compile_state
    cleanup_compile_state(torch.device("cuda"))

    rows = discover_real_kernel_rows(
        partition_model=partition_model,
        partition_filter="all",
        backend_filter=backend_filter,
    )
    cg_rows = [r for r in rows if r.spec is not None
               and r.backend in CONCRETE_BACKENDS
               and r.kernel_root is not None
               and (r.kernel_root / r.backend / "act_quant.py").exists()]

    seen_backends = set()
    backends_to_time = []
    partition_for_cg = None
    kr_for_cg = None
    for row in cg_rows:
        if row.backend not in seen_backends:
            seen_backends.add(row.backend)
            backends_to_time.append(row.backend)
            if partition_for_cg is None:
                partition_for_cg = row.partition
                kr_for_cg = row.kernel_root

    cg_results = {}
    if backends_to_time and kr_for_cg is not None:
        if _is_rank0():
            print(f"Timing CUDA graph backends: {', '.join(backends_to_time)} ...", flush=True)
        try:
            timing = _time_cudagraph_backends(
                model, tokenizer, PROMPTS, kr_for_cg, backends_to_time,
                max_new_tokens=64, warmup=warmup, repeat=repeat,
            )
            for backend, ms in timing.items():
                cg_results[backend] = (partition_for_cg, ms)
                if _is_rank0():
                    print(f"  {backend}: {ms:.1f}ms", flush=True)
        except Exception as exc:
            if _is_rank0():
                import traceback
                print(f"  FAILED: {exc}", flush=True)
                traceback.print_exc()

    return {
        "baseline_ms": baseline_ms,
        "compile_ms": compile_ms,
        "cg_results": cg_results,
        "partition_model": partition_model,
        "hardware": _hardware_info("cuda:0"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _render_markdown(report, slug):
    lines = [
        f"# Real Arithmetic Benchmark: {slug}",
        "",
        f"**Model**: {EVAL_MODEL}",
        f"**Partition model**: {report['partition_model']}",
        f"**GPU**: {report['hardware'].get('gpu', 'unknown')}",
        f"**PyTorch**: {torch.__version__}",
        f"**Date**: {report['timestamp']}",
        f"**Decode tokens**: {64 * len(PROMPTS)} ({len(PROMPTS)} prompts x 64 tokens)",
        "",
    ]

    rows = []
    rows.append(("baseline", "torch", report["baseline_ms"]))
    rows.append(("baseline", "torch.compile", report["compile_ms"]))
    for backend, (partition, ms) in report["cg_results"].items():
        rows.append((partition, backend, ms))
    rows.sort(key=lambda r: r[2])

    baseline_ms = report["baseline_ms"]
    winner = rows[0]
    lines.append(f"## Winner")
    lines.append(f"")
    lines.append(f"**{winner[0]}/{winner[1]}** ({baseline_ms / winner[2]:.3f}x vs baseline)")
    lines.append(f"Aggregate: {winner[2]:.1f}ms")
    lines.append("")
    lines.append("## Leaderboard")
    lines.append("")
    lines.append("| # | partition | backend | total (ms) | vs baseline |")
    lines.append("|---|---|---|---|---|")
    for i, (partition, backend, ms) in enumerate(rows):
        speedup = baseline_ms / ms
        lines.append(f"| {i+1} | {partition} | {backend} | {ms:.1f} | {speedup:.3f}x |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Real-weight arithmetic benchmark")
    parser.add_argument("--ckpt-path", type=Path, default=Path("checkpoints/dsv3_2-mp8"))
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-direct", action="store_true")
    parser.add_argument("--partition-model", default="dsv3_2")
    parser.add_argument("--backend", default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    hf_token = require_hf_token(args.hf_token, purpose="Real arithmetic benchmark")
    report = run(
        ckpt_path=args.ckpt_path,
        hf_token=hf_token,
        hf_direct=args.hf_direct,
        partition_model=args.partition_model,
        backend_filter=args.backend,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    if not _is_rank0():
        return

    slug = _hardware_slug("cuda:0")
    output = args.output or Path(__file__).parent / "results" / args.partition_model / f"{slug}_real.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_markdown(report, slug))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
