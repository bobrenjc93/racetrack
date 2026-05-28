"""Shared infrastructure for real-weight benchmark runners.

Functions extracted from benchmarks/gsm8k/real_bench.py and
benchmarks/arithmetic/real_bench.py to eliminate duplication.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from benchmarks.common import CONCRETE_BACKENDS, DSV3_2_CONFIG, EVAL_MODEL


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_rank0() -> bool:
    return _rank() == 0


def can_use_fused_patches(row) -> bool:
    if row.kernel_root is None or row.spec is None:
        return False
    kr = row.kernel_root or row.spec.kernel_root
    for backend in CONCRETE_BACKENDS:
        if (kr / backend / "act_quant.py").exists():
            return True
    return False


def resolve_selected_backends(row) -> dict[str, tuple[str, ...]]:
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


def load_model_and_tokenizer(
    *,
    ckpt_path: Path,
    hf_token: str,
    max_seq_len: int = 4096,
    hf_direct: bool,
    strict: bool = True,
):
    import torch.distributed as dist
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_model
    from transformers import PreTrainedTokenizerFast

    from benchmarks.gsm8k.hf_model_loader import (
        load_hf_sharded_weights,
        run_post_load_transforms,
    )
    from racetrack.models.deepseek import ModelArgs, Transformer

    world_size = _world_size()
    rank = _rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("Real benchmark requires CUDA")
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
        load_model(model, str(ckpt_file), strict=strict)
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


def build_and_capture_cudagraph(model, kernel_root, backend, prompt_len, max_seq_len):
    """Build flat decode function and capture CUDA graph.

    Returns (flat_fn, flat_cg_fn, update_bufs, static_logits, graph, static_tok).
    """
    from benchmarks.gsm8k.flat_decode import build_flat_decode

    flat_fn, flat_cg_fn, update_bufs, s_logits = build_flat_decode(
        model, kernel_root, backend=backend,
        max_seq_len=max_seq_len,
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

    return flat_fn, flat_cg_fn, update_bufs, s_logits, graph, static_tok
