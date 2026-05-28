"""Real-weight arithmetic accuracy evaluation.

Loads the DeepSeek-V3.2 checkpoint through the repository's inference model,
generates answers for the arithmetic identity prompts, and caches per-question
correctness for the latency benchmark report.

Usage:
    torchrun --standalone --nproc-per-node=8 \
      -m benchmarks.arithmetic.eval \
      --ckpt-path checkpoints/dsv3_2-mp8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

for _tv_mod in ("torchvision", "torchvision.transforms"):
    if _tv_mod not in sys.modules:
        sys.modules[_tv_mod] = None  # type: ignore[assignment]

import torch

from benchmarks.arithmetic.bench import CASES
from benchmarks.common import DSV3_2_CONFIG, EVAL_MODEL

MAX_NEW_TOKENS = 64
CACHE_PATH = Path(__file__).parent / "results" / "eval_cache.json"


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2) + "\n")


def extract_answer(text: str) -> int | None:
    numbers = re.findall(r"[+-]?\d[\d,]*", text)
    if not numbers:
        return None
    return int(numbers[-1].replace(",", ""))


def _prompt(question: str) -> str:
    return (
        "<｜begin▁of▁sentence｜>"
        + "<｜User｜>"
        + question
        + "\nReply with only the exact integer answer."
        + "<｜Assistant｜>"
    )


@torch.inference_mode()
def _generate_greedy(
    model,
    prompt_tokens: list[list[int]],
    max_new_tokens: int,
    eos_id: int,
) -> list[list[int]]:
    prompt_lens = [len(t) for t in prompt_tokens]
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full(
        (len(prompt_tokens), total_len),
        -1,
        dtype=torch.long,
        device="cuda",
    )
    for i, token_ids in enumerate(prompt_tokens):
        tokens[i, : len(token_ids)] = torch.tensor(
            token_ids,
            dtype=torch.long,
            device="cuda",
        )

    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1

    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = logits.argmax(dim=-1)
        next_token = torch.where(
            prompt_mask[:, cur_pos],
            tokens[:, cur_pos],
            next_token,
        )
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(
            ~prompt_mask[:, cur_pos],
            next_token == eos_id,
        )
        prev_pos = cur_pos
        if finished.all():
            break

    completion_tokens = []
    for i, token_ids in enumerate(tokens.tolist()):
        token_ids = token_ids[prompt_lens[i] : prompt_lens[i] + max_new_tokens]
        if eos_id in token_ids:
            token_ids = token_ids[: token_ids.index(eos_id)]
        completion_tokens.append(token_ids)
    return completion_tokens


def evaluate(
    ckpt_path: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    force: bool = False,
) -> dict:
    cache = _load_cache()
    cache_key = f"{EVAL_MODEL}:arithmetic:{len(CASES)}:{max_new_tokens}:v1"
    if not force and cache_key in cache:
        result = cache[cache_key]
        print(
            f"Using cached eval: {EVAL_MODEL} arithmetic "
            f"-> {result['accuracy_pct']}%"
        )
        return result

    import torch.distributed as dist
    from safetensors.torch import load_model
    from transformers import PreTrainedTokenizerFast

    from racetrack.models.deepseek import ModelArgs, Transformer

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")

    if not torch.cuda.is_available():
        raise RuntimeError("Arithmetic eval requires CUDA")
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)

    config = dict(DSV3_2_CONFIG)
    config["max_batch_size"] = 1
    config["max_seq_len"] = 256
    config["dtype"] = "fp8"
    args = ModelArgs(**config)

    if rank == 0:
        print(f"Loading model from {ckpt_path} (world_size={world_size}) ...")

    with torch.device("cuda"):
        model = Transformer(args)

    ckpt_file = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    load_model(model, ckpt_file)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(ckpt_path, "tokenizer.json"),
    )

    correct = 0
    case_results = []
    eos_id = 1

    for case in CASES:
        prompt_tokens = tokenizer.encode(_prompt(case.prompt))
        completion_tokens = _generate_greedy(
            model,
            [prompt_tokens],
            max_new_tokens,
            eos_id,
        )
        response = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
        predicted = extract_answer(response)
        expected = int(case.expected)
        is_correct = predicted == expected
        correct += int(is_correct)
        case_results.append(
            {
                "name": case.name,
                "prompt": case.prompt,
                "expected": case.expected,
                "predicted": None if predicted is None else str(predicted),
                "response": response,
                "correct": is_correct,
                "correctness_pct": 100.0 if is_correct else 0.0,
            }
        )
        if rank == 0:
            print(
                f"{case.name}: predicted={predicted} "
                f"expected={case.expected} correct={is_correct}"
            )

    total = len(CASES)
    accuracy = correct / total * 100.0
    result = {
        "model": EVAL_MODEL,
        "num_samples": total,
        "correct": correct,
        "accuracy_pct": round(accuracy, 1),
        "cases": case_results,
    }

    if rank == 0:
        cache[cache_key] = result
        _save_cache(cache)
        print(f"Final arithmetic accuracy: {accuracy:.1f}% ({correct}/{total})")

    del model
    torch.cuda.empty_cache()

    if world_size > 1:
        dist.destroy_process_group()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Arithmetic real-weight eval")
    parser.add_argument(
        "--ckpt-path",
        default="checkpoints/dsv3_2-mp8",
        help="Path to converted checkpoint directory",
    )
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--force", action="store_true", help="Ignore cache")
    args = parser.parse_args()

    result = evaluate(args.ckpt_path, args.max_new_tokens, args.force)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
