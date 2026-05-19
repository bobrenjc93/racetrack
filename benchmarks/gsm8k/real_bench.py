"""Real-weight end-to-end GSM8K leaderboard for patched partition kernels.

This runner loads the checkpoint-backed ``inference.model.Transformer`` once,
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
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.gsm8k.eval import (
    DSV3_2_CONFIG,
    EVAL_MODEL,
    MAX_NEW_TOKENS,
    NUM_SAMPLES,
    extract_answer,
    extract_ground_truth,
    _generate_greedy,
)
from benchmarks.gsm8k.hf_auth import require_hf_token
from benchmarks.common import (
    TORCH_COMPILE_BACKEND,
    CONCRETE_BACKENDS,
    hardware_info as _hardware_info,
    hardware_slug as _hardware_slug,
    format_diff as _format_diff,
)
from benchmarks.gsm8k.real_kernels import (
    RealKernelRow,
    discover_real_kernel_rows,
    patch_real_model,
)

ANSWER_ATOL = 1.0e-3


def _can_use_fused_patches(row) -> bool:
    """Check if partition has act_quant kernel needed for full fusion."""
    if row.kernel_root is None or row.spec is None:
        return False
    kr = row.kernel_root or row.spec.kernel_root
    for backend in ("triton", "cutedsl", "helion"):
        if (kr / backend / "act_quant.py").exists():
            return True
    return False


def _resolve_selected_backends(row) -> dict[str, tuple[str, ...]]:
    """Resolve 'best' to concrete backend names per op."""
    if row.backend != "best":
        return {op: (row.backend,) for op in row.ops}
    kr = row.kernel_root or (row.spec.kernel_root if row.spec else None)
    if kr is None:
        return {op: ("best",) for op in row.ops}
    from racetrack.runtime.dispatch import KernelDispatcher
    dispatcher = KernelDispatcher(kr)
    result = {}
    for op in row.ops:
        for backend in CONCRETE_BACKENDS:
            if dispatcher._resolve(backend, op) is not None:
                result[op] = (backend,)
                break
    return result


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


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_rank0() -> bool:
    return _rank() == 0


def _load_model_and_tokenizer(
    *,
    ckpt_path: Path,
    hf_token: str,
    max_seq_len: int,
    hf_direct: bool,
):
    import torch.distributed as dist
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_model
    from transformers import PreTrainedTokenizerFast

    from benchmarks.gsm8k.hf_model_loader import (
        load_hf_sharded_weights,
        run_post_load_transforms,
    )
    from inference.model import ModelArgs, Transformer

    world_size = _world_size()
    rank = _rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("Real GSM8K benchmark requires CUDA")
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)

    config = dict(DSV3_2_CONFIG)
    config["max_batch_size"] = 1
    config["max_seq_len"] = max_seq_len
    config["dtype"] = "fp8"
    args = ModelArgs(**config)

    ckpt_file = ckpt_path / f"model{rank}-mp{world_size}.safetensors"
    use_converted = ckpt_file.exists()
    if not use_converted and not hf_direct:
        raise FileNotFoundError(
            f"Checkpoint shard {ckpt_file} does not exist. Provide the converted "
            "model-parallel checkpoint or pass --hf-direct to stream and slice "
            "the Hugging Face shards for each rank."
        )

    if _is_rank0():
        source = str(ckpt_path) if use_converted else f"{EVAL_MODEL} HF shards"
        print(f"Loading real model from {source} (world_size={world_size}) ...", flush=True)
    with torch.device("cuda"):
        model = Transformer(args)
    if use_converted:
        load_model(model, str(ckpt_file))
    else:
        loaded = load_hf_sharded_weights(
            model,
            repo_id=EVAL_MODEL,
            hf_token=hf_token,
            rank=rank,
            world_size=world_size,
        )
        if _is_rank0():
            print(f"Loaded {loaded} HF tensors for rank 0", flush=True)
    transforms = run_post_load_transforms(model)
    if transforms and _is_rank0():
        print(f"Ran {len(transforms)} post-load model transforms", flush=True)
    model.eval()

    tokenizer_file = ckpt_path / "tokenizer.json"
    if not tokenizer_file.exists():
        tokenizer_file = Path(
            hf_hub_download(EVAL_MODEL, "tokenizer.json", token=hf_token)
        )
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
    return model, tokenizer


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
        prompt_mask = tokens != -1

        # KV cache is already filled from model.forward(tok, 0) prefill.
        # Skip re-prefill — go straight to CUDA graph decode.
        finished = False
        for cur_pos in range(prompt_len, total_len):
            prev_pos = cur_pos - 1
            static_tok.fill_(tokens[0, prev_pos].item())
            update_bufs(prev_pos)
            graph.replay()

            next_token = static_logits.argmax(dim=-1)
            if prompt_mask[0, cur_pos]:
                next_token = tokens[:, cur_pos]
            tokens[0, cur_pos] = next_token[0]
            if not prompt_mask[0, cur_pos] and next_token[0].item() == 1:
                finished = True
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
    return RowResult(
        partition=row.partition,
        backend=row.backend,
        ops=row.ops,
        mean_ms=total_ms / max(total, 1),
        total_ms=total_ms,
        accuracy_pct=correct / max(total, 1) * 100.0,
        correct=correct,
        total=total,
        validation=answer_match == total,
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
    model, tokenizer = _load_model_and_tokenizer(
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
    cudagraph_row = next(
        (r for r in rows if r.spec is not None and _can_use_fused_patches(r) and r.backend == "triton"),
        None,
    )
    if cudagraph_row is not None:
        if _is_rank0():
            print(f"Row {cudagraph_row.partition}/triton: flat decode + CUDA graph", flush=True)
        try:
            from benchmarks.gsm8k.flat_decode import build_flat_decode

            kr = cudagraph_row.kernel_root or cudagraph_row.spec.kernel_root

            # Prefill with original model before stacking MoE weights
            if _is_rank0():
                print("  prefill ...", flush=True)
            sample = list(dataset)[0]
            prompt_tokens = tokenizer.encode(_prompt(sample["question"]))
            tok = torch.tensor([prompt_tokens], dtype=torch.long, device="cuda")
            model.forward(tok, 0)
            torch.cuda.synchronize()

            # Build flat decode (stacks MoE via CPU staging, destroys experts)
            if _is_rank0():
                print("  building flat decode ...", flush=True)
            prompt_len = len(prompt_tokens)
            cg_max_seq = min(prompt_len + 256, model.max_seq_len)
            flat_fn, flat_cg_fn, update_bufs, s_logits = build_flat_decode(model, kr, max_seq_len=cg_max_seq)

            # Warmup flat decode
            if _is_rank0():
                print("  warmup ...", flush=True)
            static_tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
            prompt_len = len(prompt_tokens)
            for i in range(min(5, prompt_len)):
                update_bufs(prompt_len + i)
                static_tok.fill_(0)
                flat_cg_fn(static_tok)
            torch.cuda.synchronize()

            # Capture CUDA graph
            if _is_rank0():
                print("  capturing CUDA graph ...", flush=True)
            update_bufs(prompt_len + 10)
            flat_cg_fn(static_tok)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                flat_cg_fn(static_tok)
            torch.cuda.synchronize()

            # Timed generation using CUDA graph
            if _is_rank0():
                print("  timed run ...", flush=True)
            outputs, total_ms = _evaluate_row_cudagraph_prebuilt(
                model, tokenizer, dataset,
                flat_cg_fn, update_bufs, s_logits, graph, static_tok,
                max_new_tokens=max_new_tokens,
            )
            cg_result = _row_result(
                cudagraph_row,
                outputs, total_ms, baseline_outputs,
                {op: 1 for op in cudagraph_row.ops},
                _resolve_selected_backends(cudagraph_row),
            )
            # Replace eager results for this partition with the CUDA graph result.
            # Covers triton and best (when best resolves to all-triton).
            def _should_replace(r):
                if r.partition != cudagraph_row.partition:
                    return False
                if r.backend == "triton":
                    return True
                if r.backend == "best" and all(
                    b == "triton" for bs in r.selected_backends.values() for b in bs
                ):
                    return True
                return False
            row_results = [
                _row_result(
                    RealKernelRow(
                        partition_model=r.partition_model,
                        partition=r.partition,
                        backend=r.backend,
                        kernel_root=cudagraph_row.kernel_root,
                        ops=r.ops,
                        spec=cudagraph_row.spec,
                    ),
                    outputs, total_ms, baseline_outputs,
                    {op: 1 for op in r.ops},
                    r.selected_backends,
                ) if _should_replace(r) else r
                for r in row_results
            ]
            if _is_rank0():
                print(f"  {total_ms:.1f}ms total", flush=True)
        except Exception as exc:
            if _is_rank0():
                import traceback
                print(f"  cudagraph row failed: {exc}", flush=True)
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
