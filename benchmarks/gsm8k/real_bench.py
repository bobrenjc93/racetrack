"""Real-weight end-to-end GSM8K leaderboard for patched partition kernels.

This runner loads the checkpoint-backed ``inference.model.Transformer`` once,
evaluates the baseline, and then evaluates full-model rows where compatible
partition-local kernels are patched into the real model. Unlike
``benchmarks.gsm8k.bench``, rows here generate GSM8K answers with real weights.

Run with torchrun so the DeepSeek checkpoint is loaded with model parallelism:

    torchrun --standalone --nproc-per-node=8 -m benchmarks.gsm8k.real_bench \
        --ckpt-path checkpoints/dsv3_2-mp8 \
        --samples 50
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.gsm8k.bench import _hardware_info, _hardware_slug
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
from benchmarks.gsm8k.real_kernels import (
    RealKernelRow,
    discover_real_kernel_rows,
    patch_real_model,
)


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
    answer_match = sum(
        _answers_match(result.predicted, baseline.predicted)
        for result, baseline in zip(outputs, baseline_outputs)
    )
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
        calls=dict(calls),
        selected_backends=dict(selected_backends or {}),
    )


def _answers_match(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) < 1.0e-3


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
    if partition_model != "dsv3_2_nvfp4":
        raise ValueError("Real GSM8K partition patching currently supports dsv3_2_nvfp4 only")

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
        with patch_real_model(model, row, strict_kernel_use=True) as stats:
            outputs, total_ms = _evaluate_row(
                model,
                tokenizer,
                dataset,
                max_new_tokens=max_new_tokens,
            )
        row_result = _row_result(
            row,
            outputs,
            total_ms,
            baseline_outputs,
            stats.calls,
            stats.selected_backends,
        )
        if require_pass and not row_result.validation:
            raise RuntimeError(
                f"{row.label} failed validation: "
                f"{row_result.answer_match}/{row_result.total} extracted answers "
                f"matched baseline ({row_result.token_match}/{row_result.total} "
                "exact completions matched)"
            )
        row_results.append(row_result)

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
        key=lambda row: (not row["validation"], row["mean_ms"]),
    )
    lines = [
        f"# Real GSM8K E2E Benchmark: {slug}",
        "",
        f"**Model**: {report['model']}",
        f"**Partition model**: {report['partition_model']}",
        f"**GPU**: {hw.get('gpu', 'N/A')} x{hw.get('gpu_count', 1)}",
        f"**CUDA**: {hw.get('cuda', 'N/A')}",
        f"**PyTorch**: {hw.get('torch', 'N/A')}",
        f"**Samples**: {report['samples']}",
        f"**Max new tokens**: {report['max_new_tokens']}",
        f"**Date**: {report['timestamp']}",
        "",
        "## Leaderboard",
        "",
    ]
    headers = [
        "#",
        "partition",
        "backend",
        "ops",
        "accuracy",
        "validation",
        "answer match",
        "token match",
        "mean ms/example",
        "kernel calls",
        "selected",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for rank, row in enumerate(rows, 1):
        call_count = sum(row["calls"].values())
        ops = ", ".join(row["ops"]) if row["ops"] else "baseline"
        selected = "; ".join(
            f"{op}={'+'.join(backends)}"
            for op, backends in sorted(row.get("selected_backends", {}).items())
        ) or "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row["partition"],
                    row["backend"],
                    ops,
                    f"{row['accuracy_pct']:.1f}% ({row['correct']}/{row['total']})",
                    "pass" if row["validation"] else "fail",
                    f"{row['answer_match']}/{row['total']}",
                    f"{row['token_match']}/{row['total']}",
                    f"{row['mean_ms']:.1f}",
                    str(call_count),
                    selected,
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


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
        output = Path(__file__).parent / "results" / args.partition_model / f"{slug}-real.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_markdown(report, slug))
    print(f"Saved markdown report to {output}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved JSON report to {args.json_output}")


if __name__ == "__main__":
    main()
