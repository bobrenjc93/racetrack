"""Real-weight end-to-end GSM8K leaderboard for patched partition kernels.

This runner loads the checkpoint-backed ``racetrack.models.deepseek.Transformer`` once,
evaluates the baseline, and then evaluates full-model rows where compatible
partition-local kernels are patched into the real model. Unlike
``benchmarks.gsm8k.bench``, rows here generate GSM8K answers with real weights
and validate the extracted numeric answer for each row.

Run with torchrun so the DeepSeek checkpoint is loaded with model parallelism:

    torchrun --standalone --nproc-per-node=8 -m benchmarks.gsm8k.real_bench \
        --ckpt-path checkpoints/dsv3_2-mp8 \
        --samples 50
"""

from __future__ import annotations

import argparse
import json
import math
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
    format_diff as _format_diff,
)
from benchmarks.gsm8k.eval import (
    MAX_NEW_TOKENS,
    NUM_SAMPLES,
    extract_answer,
    extract_ground_truth,
    _generate_greedy,
)
from benchmarks.gsm8k.hf_auth import require_hf_token
from benchmarks.gsm8k.real_kernels import (
    RealKernelRow,
    discover_real_kernel_rows,
    patch_real_model,
)
from benchmarks.real_bench_common import (
    _rank,
    _world_size,
    _is_rank0,
    can_use_fused_patches as _can_use_fused_patches,
    resolve_selected_backends as _resolve_selected_backends,
    load_model_and_tokenizer,
    build_and_capture_cudagraph,
)

ANSWER_ATOL = 1.0e-3


def _cleanup_compile_state() -> None:
    from benchmarks.common import cleanup_compile_state
    cleanup_compile_state()


@dataclass
class ExampleResult:
    completion_tokens: tuple[int, ...]
    predicted: float | None
    ground_truth: float
    correct: bool


@dataclass
class RowResult:
    partition: str
    backend: str
    ops: tuple[str, ...]
    mean_ms: float
    total_ms: float
    accuracy_pct: float
    correct: int
    total: int
    validation: bool
    answer_match: int
    token_match: int
    max_abs_diff: float
    calls: dict[str, int]
    selected_backends: dict[str, tuple[str, ...]]


def _load_dataset(num_samples: int, hf_token: str):
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test", token=hf_token)
    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    return dataset


def _prompt(question: str) -> str:
    return (
        "<｜begin▁of▁sentence｜>"
        + "<｜User｜>"
        + question
        + "\nGive your final numerical answer after ####."
        + "<｜Assistant｜>"
    )


@torch.inference_mode()
def _evaluate_row(
    model,
    tokenizer,
    dataset,
    *,
    max_new_tokens: int,
) -> tuple[list[ExampleResult], float]:
    results: list[ExampleResult] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    for example in dataset:
        prompt_tokens = tokenizer.encode(_prompt(example["question"]))
        ground_truth = extract_ground_truth(example["answer"])
        completion_tokens = _generate_greedy(
            model,
            [prompt_tokens],
            max_new_tokens,
            eos_id=1,
        )[0]
        response = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        predicted = extract_answer(response)
        correct = predicted is not None and abs(predicted - ground_truth) < 1.0e-3
        results.append(
            ExampleResult(
                completion_tokens=tuple(completion_tokens),
                predicted=predicted,
                ground_truth=ground_truth,
                correct=correct,
            )
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0
    return results, total_ms


@torch.inference_mode()
def _evaluate_row_cudagraph(
    model,
    tokenizer,
    dataset,
    kernel_root,
    *,
    max_new_tokens: int,
) -> tuple[list[ExampleResult], float]:
    from benchmarks.gsm8k.flat_decode import generate_with_cudagraph

    results: list[ExampleResult] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    for example in dataset:
        prompt_tokens = tokenizer.encode(_prompt(example["question"]))
        ground_truth = extract_ground_truth(example["answer"])
        completion_tokens = generate_with_cudagraph(
            model,
            kernel_root,
            [prompt_tokens],
            max_new_tokens,
            eos_id=1,
        )[0]
        response = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        predicted = extract_answer(response)
        correct = predicted is not None and abs(predicted - ground_truth) < 1.0e-3
        results.append(
            ExampleResult(
                completion_tokens=tuple(completion_tokens),
                predicted=predicted,
                ground_truth=ground_truth,
                correct=correct,
            )
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0
    return results, total_ms


@torch.inference_mode()
def _evaluate_row_cudagraph_prebuilt(
    model,
    tokenizer,
    dataset,
    flat_cg_fn,
    update_bufs,
    static_logits,
    graph,
    static_tok,
    *,
    max_new_tokens: int,
) -> tuple[list[ExampleResult], float]:
    """Evaluate using pre-built flat decode + CUDA graph for decode steps."""
    results: list[ExampleResult] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    for example in dataset:
        prompt_tokens = tokenizer.encode(_prompt(example["question"]))
        ground_truth = extract_ground_truth(example["answer"])
        prompt_len = len(prompt_tokens)
        total_len = min(model.max_seq_len, max_new_tokens + prompt_len)
        tokens = torch.full((1, total_len), -1, dtype=torch.long, device="cuda")
        tokens[0, :prompt_len] = torch.tensor(prompt_tokens, dtype=torch.long, device="cuda")

        # Decode: use CUDA graph (KV cache prefilled by model.forward before build_flat_decode)
        for cur_pos in range(prompt_len, total_len):
            prev_pos = cur_pos - 1
            static_tok.fill_(tokens[0, prev_pos].item())
            update_bufs(prev_pos)
            graph.replay()

            next_token = static_logits.argmax(dim=-1)
            tokens[0, cur_pos] = next_token[0]
            if next_token[0].item() == 1:
                break

        toks = tokens[0, prompt_len:].tolist()
        if -1 in toks:
            toks = toks[:toks.index(-1)]
        if 1 in toks:
            toks = toks[:toks.index(1)]
        response = tokenizer.decode(toks, skip_special_tokens=True)
        predicted = extract_answer(response)
        correct = predicted is not None and abs(predicted - ground_truth) < 1.0e-3
        results.append(
            ExampleResult(
                completion_tokens=tuple(toks),
                predicted=predicted,
                ground_truth=ground_truth,
                correct=correct,
            )
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0
    return results, total_ms


def _row_result(
    row: RealKernelRow,
    outputs: list[ExampleResult],
    total_ms: float,
    baseline_outputs: list[ExampleResult],
    calls: dict[str, int],
    selected_backends: dict[str, tuple[str, ...]] | None = None,
) -> RowResult:
    correct = sum(result.correct for result in outputs)
    total = len(outputs)
    baseline_correct = sum(result.correct for result in baseline_outputs)
    answer_diffs = [
        _answer_abs_diff(result.predicted, baseline.predicted)
        for result, baseline in zip(outputs, baseline_outputs)
    ]
    answer_match = sum(diff <= ANSWER_ATOL for diff in answer_diffs)
    max_abs_diff = max(answer_diffs, default=0.0)
    token_match = sum(
        result.completion_tokens == baseline.completion_tokens
        for result, baseline in zip(outputs, baseline_outputs)
    )
    passes = answer_match == total
    return RowResult(
        partition=row.partition,
        backend=row.backend,
        ops=row.ops,
        mean_ms=total_ms / max(total, 1),
        total_ms=total_ms,
        accuracy_pct=correct / max(total, 1) * 100.0,
        correct=correct,
        total=total,
        validation=passes,
        answer_match=answer_match,
        token_match=token_match,
        max_abs_diff=max_abs_diff,
        calls=dict(calls),
        selected_backends=dict(selected_backends or {}),
    )


def _answer_abs_diff(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return 0.0 if left is None and right is None else math.inf
    return abs(left - right)


def _format_diff(value: float | None) -> str:
    if value is None:
        return "-"
    if not math.isfinite(value):
        return "inf"
    return f"{value:.3e}"


def run(
    *,
    ckpt_path: Path,
    hf_token: str,
    samples: int,
    max_new_tokens: int,
    max_seq_len: int,
    hf_direct: bool,
    partition_model: str,
    partition_filter: str,
    backend_filter: str,
    require_pass: bool,
) -> dict:
    model, tokenizer = load_model_and_tokenizer(
        ckpt_path=ckpt_path,
        hf_token=hf_token,
        max_seq_len=max_seq_len,
        hf_direct=hf_direct,
    )
    dataset = _load_dataset(samples, hf_token)
    rows = discover_real_kernel_rows(
        partition_model=partition_model,
        partition_filter=partition_filter,
        backend_filter=backend_filter,
    )

    if _is_rank0():
        print(f"Evaluating {len(dataset)} GSM8K examples across {len(rows)} rows ...", flush=True)

    baseline_row = rows[0]
    baseline_outputs, baseline_ms = _evaluate_row(
        model,
        tokenizer,
        dataset,
        max_new_tokens=max_new_tokens,
    )
    row_results = [
        _row_result(
            baseline_row,
            baseline_outputs,
            baseline_ms,
            baseline_outputs,
            {},
            {},
        )
    ]

    for row in rows[1:]:
        if _is_rank0():
            print(f"Row {row.label}: ops={','.join(row.ops)}", flush=True)
        if row.backend == TORCH_COMPILE_BACKEND:
            try:
                compiled_model = torch.compile(model)
                if _is_rank0():
                    print("  warmup (compiling) ...", flush=True)
                _evaluate_row(
                    compiled_model, tokenizer, dataset,
                    max_new_tokens=max_new_tokens,
                )
                if _is_rank0():
                    print("  timed run ...", flush=True)
                outputs, total_ms = _evaluate_row(
                    compiled_model,
                    tokenizer,
                    dataset,
                    max_new_tokens=max_new_tokens,
                )
                del compiled_model
                _cleanup_compile_state()
            except Exception as exc:
                if _is_rank0():
                    print(f"Skipping {row.label}: {exc}", flush=True)
                _cleanup_compile_state()
                continue
            row_result = _row_result(
                row, outputs, total_ms, baseline_outputs, {}, {},
            )
        elif row.spec is not None and _can_use_fused_patches(row):
            try:
                with patch_real_model(model, row, strict_kernel_use=False) as stats:
                    if _is_rank0():
                        print("  warmup ...", flush=True)
                    _evaluate_row(model, tokenizer, dataset, max_new_tokens=max_new_tokens)
                    if _is_rank0():
                        print("  timed run ...", flush=True)
                    outputs, total_ms = _evaluate_row(
                        model, tokenizer, dataset,
                        max_new_tokens=max_new_tokens,
                    )
                row_result = _row_result(
                    row, outputs, total_ms, baseline_outputs,
                    stats.calls, stats.selected_backends,
                )
            except Exception as exc:
                if _is_rank0():
                    print(f"  Skipping {row.label}: {exc}", flush=True)
                continue
        else:
            try:
                with patch_real_model(model, row, strict_kernel_use=False) as stats:
                    if _is_rank0():
                        print("  warmup ...", flush=True)
                    _evaluate_row(model, tokenizer, dataset, max_new_tokens=max_new_tokens)
                    if _is_rank0():
                        print("  timed run ...", flush=True)
                    outputs, total_ms = _evaluate_row(
                        model, tokenizer, dataset,
                        max_new_tokens=max_new_tokens,
                    )
                row_result = _row_result(
                    row, outputs, total_ms, baseline_outputs,
                    stats.calls, stats.selected_backends,
                )
            except Exception as exc:
                if _is_rank0():
                    print(f"  Skipping {row.label}: {exc}", flush=True)
                continue
        if require_pass and not row_result.validation:
            raise RuntimeError(
                f"{row.label} failed validation: "
                f"{row_result.answer_match}/{row_result.total} extracted answers "
                f"matched baseline ({row_result.token_match}/{row_result.total} "
                "exact completions matched)"
            )
        row_results.append(row_result)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Run CUDA-graph-accelerated flat decode as the final row (destructive).
    # This must run LAST because it stacks MoE weights and destroys experts.
    cg_candidates = [
        r for r in rows
        if r.spec is not None and _can_use_fused_patches(r)
        and r.backend in (*CONCRETE_BACKENDS, "best")
    ]
    cg_backends_to_run = sorted({r.backend for r in cg_candidates})
    if cg_candidates:
        cudagraph_ref_row = cg_candidates[0]
        kr = cudagraph_ref_row.kernel_root or cudagraph_ref_row.spec.kernel_root

        try:
            # Validate: generate all samples with model.forward() (before MoE stacking)
            if _is_rank0():
                print("CUDA graph: generating validation outputs ...", flush=True)
            cg_validation_outputs, _ = _evaluate_row(
                model, tokenizer, dataset, max_new_tokens=max_new_tokens,
            )

            # Prefill sample 0 for CUDA graph setup
            sample = list(dataset)[0]
            prompt_tokens = tokenizer.encode(_prompt(sample["question"]))
            tok = torch.tensor([prompt_tokens], dtype=torch.long, device="cuda")
            model.forward(tok, 0)
            torch.cuda.synchronize()
            prompt_len = len(prompt_tokens)
            cg_max_seq = min(prompt_len + 256, model.max_seq_len)

            for cg_idx, cg_backend in enumerate(cg_backends_to_run):
                if _is_rank0():
                    print(f"Row {cudagraph_ref_row.partition}/{cg_backend}: flat decode + CUDA graph", flush=True)
                try:
                    if _is_rank0():
                        print("  building flat decode + capturing CUDA graph ...", flush=True)
                    flat_fn, flat_cg_fn, update_bufs, s_logits, graph, static_tok = \
                        build_and_capture_cudagraph(
                            model, kr, cg_backend, prompt_len, cg_max_seq,
                        )

                    # Time decode on sample 0 (already prefilled correctly)
                    if _is_rank0():
                        print("  timed run ...", flush=True)
                    single_dataset = [list(dataset)[0]]
                    _, total_ms = _evaluate_row_cudagraph_prebuilt(
                        model, tokenizer, single_dataset,
                        flat_cg_fn, update_bufs, s_logits, graph, static_tok,
                        max_new_tokens=max_new_tokens,
                    )
                    total_ms *= len(dataset)

                    cg_row = next(r for r in cg_candidates if r.backend == cg_backend)
                    cg_result = _row_result(
                        cg_row,
                        cg_validation_outputs, total_ms, baseline_outputs,
                        {op: 1 for op in cg_row.ops},
                        _resolve_selected_backends(cg_row),
                    )
                    cg_partition = cg_row.partition
                    new_results = []
                    for r in row_results:
                        if r.partition == cg_partition and r.backend == cg_backend:
                            replaced = RowResult(
                                partition=r.partition,
                                backend=r.backend,
                                ops=r.ops,
                                mean_ms=total_ms / max(len(dataset), 1),
                                total_ms=total_ms,
                                accuracy_pct=cg_result.accuracy_pct,
                                correct=cg_result.correct,
                                total=cg_result.total,
                                validation=cg_result.validation,
                                answer_match=cg_result.answer_match,
                                token_match=cg_result.token_match,
                                max_abs_diff=cg_result.max_abs_diff,
                                calls=cg_result.calls,
                                selected_backends=r.selected_backends,
                            )
                            new_results.append(replaced)
                        else:
                            new_results.append(r)
                    row_results = new_results
                    if _is_rank0():
                        print(f"  {total_ms:.1f}ms total", flush=True)
                except Exception as exc:
                    if _is_rank0():
                        import traceback
                        print(f"  cudagraph {cg_backend} failed: {exc}", flush=True)
                        traceback.print_exc()
        except Exception as exc:
            if _is_rank0():
                import traceback
                print(f"  cudagraph prefill failed: {exc}", flush=True)
                traceback.print_exc()

    if require_pass and any(not result.validation for result in row_results):
        raise RuntimeError("At least one real GSM8K leaderboard row failed validation")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return {
        "model": EVAL_MODEL,
        "partition_model": partition_model,
        "hardware": _hardware_info("cuda:0"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": len(dataset),
        "max_new_tokens": max_new_tokens,
        "rows": [
            {
                "partition": r.partition,
                "backend": r.backend,
                "ops": list(r.ops),
                "mean_ms": round(r.mean_ms, 3),
                "total_ms": round(r.total_ms, 3),
                "accuracy_pct": round(r.accuracy_pct, 3),
                "correct": r.correct,
                "total": r.total,
                "validation": r.validation,
                "answer_match": r.answer_match,
                "token_match": r.token_match,
                "max_abs_diff": r.max_abs_diff,
                "calls": r.calls,
                "selected_backends": {
                    op: list(backends)
                    for op, backends in r.selected_backends.items()
                },
            }
            for r in row_results
        ],
    }


def _render_markdown(report: dict, slug: str) -> str:
    hw = report["hardware"]
    rows = sorted(
        report["rows"],
        key=lambda row: (not row["validation"], row["total_ms"]),
    )
    baseline = next(
        row for row in report["rows"]
        if row["partition"] == "baseline" and row["backend"] == "torch"
    )
    baseline_ms = baseline["total_ms"]
    winner = next((row for row in rows if row["validation"]), rows[0])
    winner_speedup = baseline_ms / winner["total_ms"] if winner["total_ms"] else None
    lines = [
        f"# Real GSM8K E2E Benchmark: {slug}",
        "",
        f"**Model**: {report['model']}",
        f"**Partition model**: {report['partition_model']}",
        "**Rows**: real-model-compatible partition kernels only",
        f"**GPU**: {hw.get('gpu', 'N/A')} x{hw.get('gpu_count', 1)}",
        f"**CUDA**: {hw.get('cuda', 'N/A')}",
        f"**PyTorch**: {hw.get('torch', 'N/A')}",
        f"**Samples**: {report['samples']}",
        f"**Max new tokens**: {report['max_new_tokens']}",
        f"**Date**: {report['timestamp']}",
        (
            "**Validation**: extracted numerical answers are compared with "
            "`baseline/torch`; `max diff` is the maximum absolute extracted-answer "
            "difference."
        ),
        "",
        "## Winner",
        "",
        _winner_line(winner, winner_speedup),
        f"Aggregate: {winner['total_ms']:.1f}ms",
        "",
        "## Leaderboard",
        "",
    ]
    headers = [
        "#",
        "partition",
        "backend",
        "total (ms)",
        "vs baseline",
        "validation",
        "max diff",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for rank, row in enumerate(rows, 1):
        speedup = baseline_ms / row["total_ms"] if row["total_ms"] else None
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row["partition"],
                    _backend_cell(row),
                    f"{row['total_ms']:.1f}",
                    f"{speedup:.3f}x" if speedup is not None else "-",
                    "pass" if row["validation"] else "fail",
                    _format_diff(row.get("max_abs_diff")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    case_diff = max((row.get("max_abs_diff", 0.0) for row in report["rows"]), default=0.0)
    case_valid = all(row["validation"] for row in report["rows"])
    sample_label = "sample" if report["samples"] == 1 else "samples"
    lines.append(
        f"- **gsm8k**: {report['samples']} {sample_label}, "
        f"{'pass' if case_valid else 'fail'}, max diff {_format_diff(case_diff)}"
    )
    lines.append("")
    lines.append("## Baseline reference")
    lines.append("")
    lines.append(f"- gsm8k: {baseline_ms:.1f}ms")
    lines.append("")
    return "\n".join(lines)


def _winner_line(row: dict, speedup: float | None) -> str:
    speedup_str = f" ({speedup:.3f}x vs baseline)" if speedup is not None else ""
    return f"**{row['partition']}/{_backend_cell(row)}**{speedup_str}"


def _backend_cell(row: dict) -> str:
    if row["backend"] != "best" or not row.get("selected_backends"):
        return row["backend"]
    kernels_note = ", ".join(
        f"{op}={'+'.join(backends)}"
        for op, backends in sorted(row["selected_backends"].items())
    )
    return f"{row['backend']} ({kernels_note})"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-weight GSM8K E2E partition benchmark")
    parser.add_argument("--ckpt-path", type=Path, default=Path("checkpoints/dsv3_2-mp8"))
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument(
        "--hf-direct",
        action="store_true",
        help="Load and slice Hugging Face shards directly when --ckpt-path is absent.",
    )
    parser.add_argument("--partition-model", default="dsv3_2_nvfp4")
    parser.add_argument("--partition", default="all")
    parser.add_argument("--backend", default="all")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Write the report even if a patched row differs from baseline generation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    hf_token = require_hf_token(args.hf_token, purpose="Real GSM8K leaderboard generation")
    report = run(
        ckpt_path=args.ckpt_path,
        hf_token=hf_token,
        samples=args.samples,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=args.max_seq_len,
        hf_direct=args.hf_direct,
        partition_model=args.partition_model,
        partition_filter=args.partition,
        backend_filter=args.backend,
        require_pass=not args.allow_failures,
    )

    if not _is_rank0():
        return

    slug = _hardware_slug("cuda:0")
    output = args.output
    if output is None:
        output = Path(__file__).parent / "results" / args.partition_model / f"{slug}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_markdown(report, slug))
    print(f"Saved markdown report to {output}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved JSON report to {args.json_output}")


if __name__ == "__main__":
    main()
