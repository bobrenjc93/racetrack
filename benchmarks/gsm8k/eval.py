"""GSM8K accuracy evaluation using a real HuggingFace model.

Downloads a small model, runs it on GSM8K test questions, extracts numerical
answers, and compares to ground truth.  Results are cached so subsequent
benchmark runs skip the expensive generation step.

Usage (standalone):
    python -m benchmarks.gsm8k.eval
    python -m benchmarks.gsm8k.eval --model Qwen/Qwen2.5-1.5B-Instruct --samples 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Block torchvision to avoid operator registration errors in dev builds
for _tv_mod in ("torchvision", "torchvision.transforms"):
    if _tv_mod not in sys.modules:
        sys.modules[_tv_mod] = None  # type: ignore[assignment]

import torch

EVAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_SAMPLES = 200
MAX_NEW_TOKENS = 512
CACHE_PATH = Path(__file__).parent / "results" / "eval_cache.json"


def _load_hf_token() -> str | None:
    env_path = Path.home() / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith("hf_token"):
                _, _, value = line.partition("=")
                return value.strip()
    return os.getenv("HF_TOKEN")


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
    model_name: str = EVAL_MODEL,
    num_samples: int = NUM_SAMPLES,
    device: str = "cuda:0",
    force: bool = False,
) -> dict:
    cache = _load_cache()
    cache_key = f"{model_name}:{num_samples}"
    if not force and cache_key in cache:
        print(f"Using cached eval: {model_name} ({num_samples} samples) "
              f"-> {cache[cache_key]['accuracy_pct']}%")
        return cache[cache_key]

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_name} ...")
    hf_token = _load_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    print("Loading GSM8K test set ...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    correct = 0
    total = len(dataset)
    print(f"Evaluating {total} GSM8K problems ...")

    for i, example in enumerate(dataset):
        question = example["question"]
        ground_truth = extract_ground_truth(example["answer"])

        messages = [
            {
                "role": "system",
                "content": (
                    "Solve this math problem step by step. "
                    "End your answer with #### followed by the numerical answer."
                ),
            },
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        predicted = extract_answer(response)

        if predicted is not None and abs(predicted - ground_truth) < 1e-3:
            correct += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{total}] accuracy so far: {correct / (i + 1) * 100:.1f}%")

    accuracy = correct / total * 100.0
    print(f"Final accuracy: {accuracy:.1f}% ({correct}/{total})")

    result = {
        "model": model_name,
        "num_samples": total,
        "correct": correct,
        "accuracy_pct": round(accuracy, 1),
    }

    cache[cache_key] = result
    _save_cache(cache)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K accuracy evaluation")
    parser.add_argument("--model", default=EVAL_MODEL)
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="Ignore cache")
    args = parser.parse_args()

    result = evaluate(args.model, args.samples, args.device, args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
