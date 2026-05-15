"""GSM8K accuracy evaluation using DeepSeek-V3.2.

Loads the full DeepSeek-V3.2 model across 8 GPUs using the official inference
code, runs it on GSM8K test questions, and reports accuracy. No cache is used;
each run evaluates the provided checkpoint.

Usage:
    torchrun --standalone --nproc-per-node=8 \
        -m benchmarks.gsm8k.eval \
        --ckpt-path checkpoints/dsv3_2-mp8 \
        --samples 200

The evaluator requires a Hugging Face token, either through --hf-token,
HF_TOKEN, or hf_token=... in ~/.env.
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

from benchmarks.gsm8k.hf_auth import require_hf_token

EVAL_MODEL = "deepseek-ai/DeepSeek-V3.2"
NUM_SAMPLES = 50
MAX_NEW_TOKENS = 4096

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


def extract_answer(text: str) -> float | None:
    match = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    numbers = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    return None


def extract_ground_truth(answer_text: str) -> float:
    match = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", answer_text)
    if match:
        return float(match.group(1).replace(",", ""))
    raise ValueError(f"Could not extract ground truth from: {answer_text}")


def evaluate(
    ckpt_path: str,
    num_samples: int = NUM_SAMPLES,
    max_new_tokens: int = MAX_NEW_TOKENS,
    hf_token: str | None = None,
) -> dict:
    hf_token = require_hf_token(hf_token, purpose="GSM8K accuracy evaluation")
    ckpt_dir = Path(ckpt_path)
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory {ckpt_dir} does not exist. Convert or mount the "
            "DeepSeek-V3.2 model-parallel checkpoint before running GSM8K eval."
        )

    import torch.distributed as dist
    from datasets import load_dataset
    from safetensors.torch import load_model
    from transformers import PreTrainedTokenizerFast

    from inference.model import ModelArgs, Transformer

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ckpt_file = ckpt_dir / f"model{rank}-mp{world_size}.safetensors"
    tokenizer_file = ckpt_dir / "tokenizer.json"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint shard {ckpt_file} does not exist")
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer file {tokenizer_file} does not exist")

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)

    config = dict(DSV3_2_CONFIG)
    config["max_batch_size"] = 1
    config["max_seq_len"] = 4096
    config["dtype"] = "fp8"
    args = ModelArgs(**config)

    if rank == 0:
        print(f"Loading model from {ckpt_path} (world_size={world_size}) ...")

    with torch.device("cuda"):
        model = Transformer(args)

    load_model(model, str(ckpt_file))

    if rank == 0:
        print("Model loaded. Loading tokenizer ...")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
    )

    if rank == 0:
        print("Loading GSM8K test set ...")

    dataset = load_dataset("openai/gsm8k", "main", split="test", token=hf_token)
    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    from inference.model import Transformer as _  # noqa: ensure generate can find model

    correct = 0
    total = len(dataset)

    if rank == 0:
        print(f"Evaluating {total} GSM8K problems ...")

    for i, example in enumerate(dataset):
        question = example["question"]
        ground_truth = extract_ground_truth(example["answer"])

        prompt = (
            "<｜begin▁of▁sentence｜>"
            + "<｜User｜>"
            + question
            + "\nGive your final numerical answer after ####."
            + "<｜Assistant｜>"
        )
        prompt_tokens = tokenizer.encode(prompt)
        eos_id = 1
        completion_tokens = _generate_greedy(
            model, [prompt_tokens], max_new_tokens, eos_id,
        )

        response = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
        predicted = extract_answer(response)

        if predicted is not None and abs(predicted - ground_truth) < 1e-3:
            correct += 1

        if rank == 0 and i == 0:
            print(f"  [sample response] {response[:300]}", flush=True)
            print(f"  predicted={predicted}, truth={ground_truth}", flush=True)

        if rank == 0 and (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{total}] accuracy so far: {correct / (i + 1) * 100:.1f}%")

    accuracy = correct / total * 100.0
    if rank == 0:
        print(f"Final accuracy: {accuracy:.1f}% ({correct}/{total})")

    result = {
        "model": EVAL_MODEL,
        "num_samples": total,
        "correct": correct,
        "accuracy_pct": round(accuracy, 1),
    }

    del model
    torch.cuda.empty_cache()

    if world_size > 1:
        dist.destroy_process_group()

    return result


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
        (len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda",
    )
    for i, t in enumerate(prompt_tokens):
        tokens[i, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1

    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = logits.argmax(dim=-1)
        next_token = torch.where(
            prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token,
        )
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(
            ~prompt_mask[:, cur_pos], next_token == eos_id,
        )
        prev_pos = cur_pos
        if finished.all():
            break

    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i] : prompt_lens[i] + max_new_tokens]
        if eos_id in toks:
            toks = toks[: toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K accuracy evaluation")
    parser.add_argument(
        "--ckpt-path",
        default="checkpoints/dsv3_2-mp8",
        help="Path to converted checkpoint directory",
    )
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token. Falls back to HF_TOKEN or hf_token=... in ~/.env.",
    )
    args = parser.parse_args()

    result = evaluate(
        args.ckpt_path,
        args.samples,
        args.max_new_tokens,
        hf_token=args.hf_token,
    )

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
