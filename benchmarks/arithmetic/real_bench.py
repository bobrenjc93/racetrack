"""Real-weight arithmetic benchmark with flat_decode + CUDA graph.

Mirrors GSM8K real_bench: loads the real DeepSeek-V3.2 checkpoint, runs
all partitions/backends in eager mode, then CUDA graph decode rows for
partitions with act_quant.

Usage:
    torchrun --standalone --nproc-per-node=8 -m benchmarks.arithmetic.real_bench \
        --hf-direct --partition-model dsv3_2 --backend all
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.common import (
    EVAL_MODEL,
    TORCH_COMPILE_BACKEND,
    CONCRETE_BACKENDS,
    hardware_info as _hardware_info,
    hardware_slug as _hardware_slug,
    cleanup_compile_state,
)
from benchmarks.gsm8k.eval import _generate_greedy
from benchmarks.gsm8k.hf_auth import require_hf_token
from benchmarks.gsm8k.real_kernels import (
    discover_real_kernel_rows,
    patch_real_model,
)
from benchmarks.real_bench_common import (
    _is_rank0,
    can_use_fused_patches as _can_use_fused_patches,
    resolve_selected_backends as _resolve_selected_backends,
    load_model_and_tokenizer,
    build_and_capture_cudagraph,
)


PROMPTS = [
    "1 * 213813290183291 =",
    "0 + 759002341987123 =",
    "480129381029381 - 0 =",
    "98273465000123 / 1 =",
]
MAX_NEW_TOKENS = 64


@torch.inference_mode()
def _time_generation(model, tokenizer, prompts, *, max_new_tokens, warmup, repeat):
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


@dataclass
class RowResult:
    partition: str
    backend: str
    total_ms: float
    selected_backends: dict


def run(*, ckpt_path, hf_token, hf_direct, partition_model, backend_filter, warmup, repeat):
    model, tokenizer = load_model_and_tokenizer(
        ckpt_path=ckpt_path, hf_token=hf_token, max_seq_len=512,
        hf_direct=hf_direct, strict=False,
    )

    if _is_rank0():
        print("Timing baseline ...", flush=True)
    baseline_ms = _time_generation(model, tokenizer, PROMPTS,
                                    max_new_tokens=MAX_NEW_TOKENS, warmup=2, repeat=3)
    row_results = [RowResult("baseline", "torch", baseline_ms, {})]

    if _is_rank0():
        print("Timing torch.compile ...", flush=True)
    compiled = torch.compile(model)
    compile_ms = _time_generation(compiled, tokenizer, PROMPTS,
                                   max_new_tokens=MAX_NEW_TOKENS, warmup=2, repeat=3)
    row_results.append(RowResult("baseline", TORCH_COMPILE_BACKEND, compile_ms, {}))
    del compiled
    cleanup_compile_state(torch.device("cuda"))

    rows = discover_real_kernel_rows(
        partition_model=partition_model,
        partition_filter="all",
        backend_filter=backend_filter,
    )

    for row in rows[1:]:
        if row.backend == TORCH_COMPILE_BACKEND:
            continue
        if row.spec is None:
            continue
        if _is_rank0():
            print(f"Row {row.partition}/{row.backend}: {len(row.ops)} ops", flush=True)
        try:
            with patch_real_model(model, row, strict_kernel_use=False):
                if _is_rank0():
                    print("  warmup ...", flush=True)
                _time_generation(model, tokenizer, PROMPTS,
                                  max_new_tokens=MAX_NEW_TOKENS, warmup=1, repeat=1)
                if _is_rank0():
                    print("  timed run ...", flush=True)
                ms = _time_generation(model, tokenizer, PROMPTS,
                                       max_new_tokens=MAX_NEW_TOKENS, warmup=0, repeat=3)
            row_results.append(RowResult(
                row.partition, row.backend, ms, _resolve_selected_backends(row),
            ))
            if _is_rank0():
                print(f"  {ms:.1f}ms", flush=True)
        except Exception as exc:
            if _is_rank0():
                print(f"  Skipping: {exc}", flush=True)

    cg_candidates = [
        r for r in rows
        if r.spec is not None and _can_use_fused_patches(r)
        and r.backend in CONCRETE_BACKENDS
    ]
    cg_backends = sorted({r.backend for r in cg_candidates})

    if cg_candidates:
        ref_row = cg_candidates[0]
        kr = ref_row.kernel_root or ref_row.spec.kernel_root

        try:
            if _is_rank0():
                print("CUDA graph: prefill ...", flush=True)
            first_prompt = tokenizer.encode(PROMPTS[0])
            tok = torch.tensor([first_prompt], dtype=torch.long, device="cuda")
            model.forward(tok, 0)
            torch.cuda.synchronize()
            prompt_len = len(first_prompt)
            max_seq = prompt_len + MAX_NEW_TOKENS
            n_decode = MAX_NEW_TOKENS * len(PROMPTS)

            for cg_backend in cg_backends:
                if _is_rank0():
                    print(f"Row {ref_row.partition}/{cg_backend}: flat decode + CUDA graph", flush=True)
                try:
                    flat_fn, flat_cg_fn, update_bufs, s_logits, graph, static_tok = \
                        build_and_capture_cudagraph(
                            model, kr, cg_backend, prompt_len, max_seq,
                        )

                    for _ in range(warmup):
                        for i in range(n_decode):
                            update_bufs(prompt_len + (i % MAX_NEW_TOKENS))
                            graph.replay()
                    torch.cuda.synchronize()

                    times = []
                    for _ in range(repeat):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        for i in range(n_decode):
                            update_bufs(prompt_len + (i % MAX_NEW_TOKENS))
                            graph.replay()
                        end.record()
                        torch.cuda.synchronize()
                        times.append(start.elapsed_time(end))
                    ms = sum(times) / len(times)

                    cg_partition = ref_row.partition
                    new_results = []
                    for r in row_results:
                        if r.partition == cg_partition and r.backend == cg_backend:
                            new_results.append(RowResult(r.partition, r.backend, ms, r.selected_backends))
                        else:
                            new_results.append(r)
                    row_results = new_results
                    if _is_rank0():
                        print(f"  {ms:.1f}ms", flush=True)
                except Exception as exc:
                    if _is_rank0():
                        print(f"  {cg_backend} failed: {exc}", flush=True)
        except Exception as exc:
            if _is_rank0():
                import traceback
                print(f"  CUDA graph prefill failed: {exc}", flush=True)
                traceback.print_exc()

    return {
        "rows": row_results,
        "baseline_ms": baseline_ms,
        "partition_model": partition_model,
        "hardware": _hardware_info("cuda:0"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _format_backend(r):
    return r.backend


def _render_markdown(report, slug):
    rows = sorted(report["rows"], key=lambda r: r.total_ms)
    baseline_ms = report["baseline_ms"]
    winner = rows[0]

    lines = [
        f"# Real Arithmetic Benchmark: {slug}",
        "",
        f"**Model**: {EVAL_MODEL}",
        f"**Partition model**: {report['partition_model']}",
        f"**GPU**: {report['hardware'].get('gpu', 'unknown')}",
        f"**CUDA**: {report['hardware'].get('cuda', 'unknown')}",
        f"**PyTorch**: {torch.__version__}",
        f"**Date**: {report['timestamp']}",
        f"**Prompts**: {len(PROMPTS)} arithmetic identities, {MAX_NEW_TOKENS} decode tokens each",
        "",
        "## Winner",
        "",
        f"**{winner.partition}/{_format_backend(winner)}** ({baseline_ms / winner.total_ms:.3f}x vs baseline)",
        f"Aggregate: {winner.total_ms:.1f}ms",
        "",
        "## Leaderboard",
        "",
        "| # | partition | backend | total (ms) | vs baseline |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows):
        speedup = baseline_ms / r.total_ms
        lines.append(f"| {i+1} | {r.partition} | {_format_backend(r)} | {r.total_ms:.1f} | {speedup:.3f}x |")
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
    output = args.output or Path(__file__).parent / "results" / args.partition_model / f"{slug}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_markdown(report, slug))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
