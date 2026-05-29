from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from racetrack.runtime.dispatch import KernelDispatcher
from racetrack.runtime import torch_ops


CONCRETE_BACKENDS = ("triton", "cutedsl", "helion")


@dataclass(frozen=True)
class RealisticShape:
    name: str
    source: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    hc_mult: int = 1
    hc_eps: float = 1.0e-5
    rms_norm_eps: float = 1.0e-6
    rope_base: float = 10000.0
    seed: int = 1234


@dataclass
class DistBenchResult:
    model: str
    backend: str
    status: str
    world_size: int
    layers: int
    tokens: int
    dtype: str
    mean_ms: float
    min_ms: float
    max_ms: float
    tokens_per_second: float
    max_abs_diff: float | None
    max_rel_diff: float | None
    peak_mem_gib: float
    ok: bool


def realistic_shape(model: str) -> RealisticShape:
    key = model.lower().replace("-", "_")
    if key == "dsv3_2":
        return RealisticShape(
            name="dsv3_2",
            source="DeepSeek V3.2 NVFP4-like MLA/MoE dimensions from vLLM PR 38595",
            vocab_size=129280,
            hidden_size=7168,
            num_layers=61,
            num_attention_heads=128,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            moe_intermediate_size=2048,
            n_routed_experts=256,
            num_experts_per_tok=8,
            seed=38595,
        )
    raise KeyError(f"Unknown realistic model {model!r}")


def _partition_root(model: str) -> Path:
    root = Path(__file__).resolve().parents[1] / "partitions" / model
    partitions = sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "model.py").exists()
    )
    if not partitions:
        raise FileNotFoundError(f"No partition directory found under {root}")
    return partitions[0]


def _dtype(name: str) -> torch.dtype:
    if name in {"auto", "bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise KeyError(f"Unknown dtype {name!r}")


def _make_weight(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    scale: float = 0.01,
) -> torch.Tensor:
    weight = torch.empty(shape, device=device, dtype=dtype)
    weight.normal_(mean=0.0, std=scale, generator=generator)
    return weight


class ShardedRealisticBlock:
    def __init__(
        self,
        shape: RealisticShape,
        *,
        device: torch.device,
        dtype: torch.dtype,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        if shape.num_attention_heads % world_size != 0:
            raise ValueError("num_attention_heads must divide world_size")
        if shape.n_routed_experts % world_size != 0:
            raise ValueError("n_routed_experts must divide world_size")
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.rank = rank
        self.world_size = world_size
        self.local_heads = shape.num_attention_heads // world_size
        self.qk_head_dim = shape.qk_nope_head_dim + shape.qk_rope_head_dim
        self.local_q_dim = self.local_heads * self.qk_head_dim
        self.local_kv_dim = self.local_heads * (
            shape.qk_nope_head_dim + shape.v_head_dim
        )
        self.local_v_dim = self.local_heads * shape.v_head_dim
        self.local_experts = shape.n_routed_experts // world_size
        self.backend = backend
        self.dispatcher = None
        if backend != "torch":
            self.dispatcher = KernelDispatcher(_partition_root(shape.name) / "kernels")

        generator = torch.Generator(device=device)
        generator.manual_seed(shape.seed + rank)
        shared_generator = torch.Generator(device=device)
        shared_generator.manual_seed(shape.seed + 100000)
        qkv_out = shape.q_lora_rank + shape.kv_lora_rank + shape.qk_rope_head_dim
        self.w_qkv_a = _make_weight(
            (qkv_out, shape.hidden_size),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        self.q_norm = torch.ones(shape.q_lora_rank, device=device, dtype=dtype)
        self.kv_norm = torch.ones(shape.kv_lora_rank, device=device, dtype=dtype)
        self.w_q_b = _make_weight(
            (self.local_q_dim, shape.q_lora_rank),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        self.w_kv_b = _make_weight(
            (self.local_kv_dim, shape.kv_lora_rank),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        self.w_o = _make_weight(
            (shape.hidden_size, self.local_v_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        self.w_gate = _make_weight(
            (shape.n_routed_experts, shape.hidden_size),
            device=device,
            dtype=dtype,
            generator=shared_generator,
        )

        # Store the full expert-parallel shard for this rank, while reusing
        # one layer's weights across the requested layer count.
        self.w13 = _make_weight(
            (
                self.local_experts,
                2 * shape.moe_intermediate_size,
                shape.hidden_size,
            ),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        self.w2 = _make_weight(
            (
                self.local_experts,
                shape.hidden_size,
                shape.moe_intermediate_size,
            ),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        if shape.hc_mult > 1:
            hc_dim = shape.hc_mult * shape.hidden_size
            self.hc_attn_fn = _make_weight(
                (shape.hc_mult, hc_dim),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            self.hc_ffn_fn = _make_weight(
                (shape.hc_mult, hc_dim),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            self.hc_head_scale = torch.ones(
                shape.hc_mult, device=device, dtype=torch.float32
            )
            self.hc_head_base = torch.zeros(
                shape.hc_mult, device=device, dtype=torch.float32
            )

    def _fused_norm_rope(
        self,
        q_c: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fallback = torch_ops.fused_norm_rope
        if self.dispatcher is None:
            return fallback(
                q_c,
                self.q_norm,
                kv_c,
                self.kv_norm,
                k_pe,
                positions,
                eps=self.shape.rms_norm_eps,
                rope_base=self.shape.rope_base,
            )
        return self.dispatcher.call(
            "fused_norm_rope",
            fallback,
            q_c,
            self.q_norm,
            kv_c,
            self.kv_norm,
            k_pe,
            positions,
            eps=self.shape.rms_norm_eps,
            rope_base=self.shape.rope_base,
        )

    def _hc_head(self, x: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        fallback = torch_ops.hc_head
        if self.dispatcher is None:
            return fallback(
                x,
                fn,
                self.hc_head_scale,
                self.hc_head_base,
                rms_norm_eps=self.shape.rms_norm_eps,
                hc_eps=self.shape.hc_eps,
            )
        return self.dispatcher.call(
            "hc_head",
            fallback,
            x,
            fn,
            self.hc_head_scale,
            self.hc_head_base,
            rms_norm_eps=self.shape.rms_norm_eps,
            hc_eps=self.shape.hc_eps,
        )

    def _attention(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        shape = self.shape
        qkv = F.linear(hidden, self.w_qkv_a)
        q_c, kv_c, k_pe = qkv.split(
            [shape.q_lora_rank, shape.kv_lora_rank, shape.qk_rope_head_dim],
            dim=-1,
        )
        q_c, kv_c, k_pe = self._fused_norm_rope(q_c, kv_c, k_pe, positions)
        tokens = hidden.shape[0]
        q = F.linear(q_c, self.w_q_b).view(tokens, self.local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split(
            [shape.qk_nope_head_dim, shape.qk_rope_head_dim],
            dim=-1,
        )
        q_pe = torch_ops.apply_rope(
            q_pe,
            positions,
            rotary_dim=shape.qk_rope_head_dim,
            base=shape.rope_base,
        )
        q = torch.cat((q_nope, q_pe), dim=-1)
        kv = F.linear(kv_c, self.w_kv_b).view(
            tokens,
            self.local_heads,
            shape.qk_nope_head_dim + shape.v_head_dim,
        )
        k_nope, v = kv.split([shape.qk_nope_head_dim, shape.v_head_dim], dim=-1)
        k = torch.cat((k_nope, k_pe.unsqueeze(1).expand(-1, self.local_heads, -1)), dim=-1)
        local = torch_ops.causal_attention(q, k, v)
        out = F.linear(local.reshape(tokens, self.local_v_dim), self.w_o)
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    def _moe(self, hidden: torch.Tensor) -> torch.Tensor:
        shape = self.shape
        router_logits = F.linear(hidden.float(), self.w_gate.float())
        topk_logits, topk_ids = torch.topk(
            router_logits,
            shape.num_experts_per_tok,
            dim=-1,
        )
        topk_weights = torch.softmax(topk_logits, dim=-1).to(hidden.dtype)
        local = torch.zeros_like(hidden)
        expert_start = self.rank * self.local_experts
        expert_end = expert_start + self.local_experts
        for slot in range(shape.num_experts_per_tok):
            expert_ids = topk_ids[:, slot]
            local_mask = (expert_ids >= expert_start) & (expert_ids < expert_end)
            if not bool(local_mask.any()):
                continue
            local_ids = (expert_ids[local_mask] - expert_start).long()
            local_hidden = hidden[local_mask]
            gate_up = torch.bmm(
                self.w13[local_ids],
                local_hidden.unsqueeze(-1),
            ).squeeze(-1)
            gate, up = gate_up.chunk(2, dim=-1)
            expert = torch.bmm(
                self.w2[local_ids],
                torch_ops.swiglu(gate, up).unsqueeze(-1),
            ).squeeze(-1)
            local[local_mask] = local[local_mask] + expert * topk_weights[
                local_mask,
                slot,
            ].unsqueeze(-1)
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        return local

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        shape = self.shape
        if shape.hc_mult > 1:
            residual = hidden
            streams = hidden.unsqueeze(1).expand(-1, shape.hc_mult, -1).contiguous()
            hidden = self._hc_head(streams, self.hc_attn_fn)
            hidden = torch_ops.rms_norm(
                hidden,
                torch.ones(shape.hidden_size, device=self.device, dtype=self.dtype),
                shape.rms_norm_eps,
            )
            hidden = residual + self._attention(hidden, positions)
            residual = hidden
            streams = hidden.unsqueeze(1).expand(-1, shape.hc_mult, -1).contiguous()
            hidden = self._hc_head(streams, self.hc_ffn_fn)
            hidden = torch_ops.rms_norm(
                hidden,
                torch.ones(shape.hidden_size, device=self.device, dtype=self.dtype),
                shape.rms_norm_eps,
            )
            return residual + self._moe(hidden)

        residual = hidden
        hidden = torch_ops.rms_norm(
            hidden,
            torch.ones(shape.hidden_size, device=self.device, dtype=self.dtype),
            shape.rms_norm_eps,
        )
        hidden = residual + self._attention(hidden, positions)
        residual = hidden
        hidden = torch_ops.rms_norm(
            hidden,
            torch.ones(shape.hidden_size, device=self.device, dtype=self.dtype),
            shape.rms_norm_eps,
        )
        return residual + self._moe(hidden)

    def best_summary(self) -> str:
        if self.dispatcher is None:
            return "native"
        return self.dispatcher.best_summary()


class ShardedRealisticModel:
    def __init__(
        self,
        shape: RealisticShape,
        *,
        device: torch.device,
        dtype: torch.dtype,
        rank: int,
        world_size: int,
        backend: str,
        layers: int,
    ) -> None:
        self.shape = shape
        self.layers = layers
        self.block = ShardedRealisticBlock(
            shape,
            device=device,
            dtype=dtype,
            rank=rank,
            world_size=world_size,
            backend=backend,
        )

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        for _ in range(self.layers):
            hidden = self.block.forward(hidden, positions)
        return hidden

    def best_summary(self) -> str:
        return self.block.best_summary()


def _time_model(
    model: ShardedRealisticModel,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    out = None
    for _ in range(warmup):
        out = model.forward(hidden, positions)
    dist.barrier(device_ids=[device.index or 0])
    torch.cuda.synchronize(device)
    times: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = model.forward(hidden, positions)
        end.record()
        torch.cuda.synchronize(device)
        elapsed = torch.tensor([start.elapsed_time(end)], device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        times.append(float(elapsed.item()))
    assert out is not None
    return out, times


def _sync_best_selection(model: ShardedRealisticModel, device: torch.device) -> None:
    # The 'best' dispatcher times candidate kernels independently on each rank,
    # so per-rank CUDA jitter can yield divergent winners for the same op and a
    # nondeterministic, last-writer-wins best.json. Broadcast rank 0's resolved
    # selection map to every rank so the timed measurement, the reported
    # best_summary, and the persisted cache all agree on one set of winners.
    dispatcher = model.block.dispatcher
    if dispatcher is None:
        return
    payload = [dispatcher._best, dispatcher._best_ops] if dist.get_rank() == 0 else [None, None]
    dist.broadcast_object_list(payload, src=0, device=device)
    best, best_ops = payload
    # _best keys embed the source rank's device string (e.g. 'cuda:0'); rewrite
    # it to this rank's device so the cached winner is actually hit here instead
    # of silently re-timing and diverging again on the receiving rank.
    src_device = "'cuda:0'"
    dst_device = f"'{device}'"
    dispatcher._best = {key.replace(src_device, dst_device): value for key, value in best.items()}
    dispatcher._best_ops = {op: set(backends) for op, backends in best_ops.items()}


def _run_backend(
    shape: RealisticShape,
    *,
    backend: str,
    layers: int,
    tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    world_size: int,
    warmup: int,
    repeat: int,
    baseline_out: torch.Tensor | None,
    atol: float,
    rtol: float,
) -> tuple[DistBenchResult, torch.Tensor]:
    os.environ["RACETRACK_KERNEL_BACKEND"] = backend
    generator = torch.Generator(device=device)
    generator.manual_seed(shape.seed)
    hidden = torch.empty((tokens, shape.hidden_size), device=device, dtype=dtype)
    hidden.normal_(mean=0.0, std=0.01, generator=generator)
    positions = torch.arange(tokens, device=device, dtype=torch.long)
    model = ShardedRealisticModel(
        shape,
        device=device,
        dtype=dtype,
        rank=rank,
        world_size=world_size,
        backend=backend,
        layers=layers,
    )
    if backend == "best":
        # Run one forward pass so every rank's dispatcher resolves and caches a
        # per-op winner, then overwrite all ranks with rank 0's selection before
        # the timed measurement so reported timings reflect a consistent config.
        model.forward(hidden, positions)
        torch.cuda.synchronize(device)
        _sync_best_selection(model, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output, times = _time_model(
        model,
        hidden,
        positions,
        warmup=warmup,
        repeat=repeat,
        device=device,
    )
    diff = None
    rel_diff = None
    ok = True
    if baseline_out is not None:
        local_diff = (output.float() - baseline_out.float()).abs().max()
        dist.all_reduce(local_diff, op=dist.ReduceOp.MAX)
        diff = float(local_diff.item())
        local_ref = baseline_out.float().abs().max()
        dist.all_reduce(local_ref, op=dist.ReduceOp.MAX)
        ref_scale = max(float(local_ref.item()), 1.0e-12)
        rel_diff = diff / ref_scale
        ok = diff <= atol + rtol * ref_scale
    mean_ms = sum(times) / len(times)
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_tensor = torch.tensor([peak], device=device)
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    status = "native"
    if backend == "best":
        status = model.best_summary()
    result = DistBenchResult(
        model=shape.name,
        backend=backend,
        status=status,
        world_size=world_size,
        layers=layers,
        tokens=tokens,
        dtype=str(dtype).replace("torch.", ""),
        mean_ms=mean_ms,
        min_ms=min(times),
        max_ms=max(times),
        tokens_per_second=tokens * layers / (mean_ms / 1000.0),
        max_abs_diff=diff,
        max_rel_diff=rel_diff,
        peak_mem_gib=float(peak_tensor.item()),
        ok=ok,
    )
    return result, output


def _best_result(candidates: list[DistBenchResult]) -> DistBenchResult:
    ok_candidates = [result for result in candidates if result.ok]
    best = min(ok_candidates or candidates, key=lambda result: result.mean_ms)
    status = best.status if best.backend == "best" else f"pure={best.backend}"
    return DistBenchResult(
        model=best.model,
        backend="best",
        status=status,
        world_size=best.world_size,
        layers=best.layers,
        tokens=best.tokens,
        dtype=best.dtype,
        mean_ms=best.mean_ms,
        min_ms=best.min_ms,
        max_ms=best.max_ms,
        tokens_per_second=best.tokens_per_second,
        max_abs_diff=best.max_abs_diff,
        max_rel_diff=best.max_rel_diff,
        peak_mem_gib=best.peak_mem_gib,
        ok=best.ok,
    )


def _print_results(results: list[DistBenchResult]) -> None:
    if dist.get_rank() != 0:
        return
    headers = [
        "model",
        "backend",
        "status",
        "gpus",
        "layers",
        "tokens",
        "mean_ms",
        "tok*layer/s",
        "diff",
        "rel",
        "peak_gib",
        "ok",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result.model,
                result.backend,
                result.status,
                str(result.world_size),
                str(result.layers),
                str(result.tokens),
                f"{result.mean_ms:.3f}",
                f"{result.tokens_per_second:.1f}",
                "-" if result.max_abs_diff is None else f"{result.max_abs_diff:.3e}",
                "-" if result.max_rel_diff is None else f"{result.max_rel_diff:.3e}",
                f"{result.peak_mem_gib:.2f}",
                "yes" if result.ok else "no",
            ]
        )
    widths = [
        max(len(str(row[i])) for row in ([headers] + rows))
        for i in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a realistic-shape sharded synthetic DeepSeek benchmark with torchrun."
    )
    parser.add_argument("--model", default="dsv3_2", choices=("dsv3_2",))
    parser.add_argument("--backend", default="all", choices=("torch", "triton", "cutedsl", "helion", "best", "all"))
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--layers", default="realistic", help="'realistic' or an integer layer count")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--atol", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1.0e-1)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def _run_shape(
    args: argparse.Namespace,
    *,
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    world_size: int,
) -> list[DistBenchResult]:
    shape = realistic_shape(model_name)
    layers = shape.num_layers if args.layers == "realistic" else int(args.layers)
    if rank == 0:
        print(
            f"Running {shape.name} realistic shape on {world_size} GPUs: "
            f"hidden={shape.hidden_size}, layers={layers}, heads={shape.num_attention_heads}, "
            f"experts={shape.n_routed_experts}, topk={shape.num_experts_per_tok}, tokens={args.tokens}"
        )
        print("Note: weights are synthetic, sharded, and layer-reused; no real checkpoint is loaded.")

    baseline, baseline_out = _run_backend(
        shape,
        backend="torch",
        layers=layers,
        tokens=args.tokens,
        dtype=dtype,
        device=device,
        rank=rank,
        world_size=world_size,
        warmup=max(0, min(args.warmup, 1)),
        repeat=1,
        baseline_out=None,
        atol=args.atol,
        rtol=args.rtol,
    )
    results: list[DistBenchResult] = []
    if args.backend == "torch":
        results.append(baseline)
    else:
        backends = (
            list(CONCRETE_BACKENDS)
            if args.backend == "all"
            else [args.backend]
        )
        if args.backend == "all":
            candidates: list[DistBenchResult] = []
            for backend in backends:
                result, _ = _run_backend(
                    shape,
                    backend=backend,
                    layers=layers,
                    tokens=args.tokens,
                    dtype=dtype,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    baseline_out=baseline_out,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                results.append(result)
                candidates.append(result)
            mixed, _ = _run_backend(
                shape,
                backend="best",
                layers=layers,
                tokens=args.tokens,
                dtype=dtype,
                device=device,
                rank=rank,
                world_size=world_size,
                warmup=args.warmup,
                repeat=args.repeat,
                baseline_out=baseline_out,
                atol=args.atol,
                rtol=args.rtol,
            )
            candidates.append(mixed)
            results.append(_best_result(candidates))
        else:
            result, _ = _run_backend(
                shape,
                backend=args.backend,
                layers=layers,
                tokens=args.tokens,
                dtype=dtype,
                device=device,
                rank=rank,
                world_size=world_size,
                warmup=args.warmup,
                repeat=args.repeat,
                baseline_out=baseline_out if args.backend != "torch" else None,
                atol=args.atol,
                rtol=args.rtol,
            )
            results.append(result)
    return results


def main() -> None:
    if "RANK" not in os.environ:
        raise SystemExit(
            "Run with torchrun, e.g. torchrun --standalone --nproc-per-node=8 "
            "-m racetrack.realistic_bench --model dsv3_2 --backend all"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        args = parse_args()
        dtype = _dtype(args.dtype)
        results: list[DistBenchResult] = []
        results.extend(
            _run_shape(
                args,
                model_name=args.model,
                dtype=dtype,
                device=device,
                rank=rank,
                world_size=world_size,
            )
        )
        _print_results(results)
        if rank == 0 and args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps([asdict(result) for result in results], indent=2))
        if not all(result.ok for result in results):
            raise SystemExit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
