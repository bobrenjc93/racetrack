from __future__ import annotations

import importlib.util
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

MODEL_NAME = "dsv3_2"
PARTITION_HASH = "a1f6d7e2"
PARTITION_NOTES = (
    "Fuses q/kv RMSNorm and RoPE at the MLA/indexer boundary, matching the "
    "shape of the monolithic attention path introduced in vLLM PR 38595."
)

Fallback = Callable[..., Any]


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x_float = x.float()
    scale = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    return (x_float * scale * weight.float()).to(orig_dtype)


def rope_cache(
    positions: torch.Tensor,
    rotary_dim: int,
    *,
    base: float = 10000.0,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    device = positions.device
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, half, device=device, dtype=torch.float32) / max(half, 1))
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int | None = None,
    base: float = 10000.0,
) -> torch.Tensor:
    rotary_dim = x.shape[-1] if rotary_dim is None else rotary_dim
    if rotary_dim == 0:
        return x
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"rotary_dim {rotary_dim} exceeds tensor dim {x.shape[-1]}")
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")

    cos, sin = rope_cache(positions, rotary_dim, base=base, dtype=x.dtype)
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    while cos.dim() < x1.dim():
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    if x_pass.numel() == 0:
        return rotated
    return torch.cat((rotated, x_pass), dim=-1)


def fused_norm_rope(
    q_c: torch.Tensor,
    q_weight: torch.Tensor,
    kv_c: torch.Tensor,
    kv_weight: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
    *,
    eps: float,
    rope_base: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        rms_norm(q_c, q_weight, eps),
        rms_norm(kv_c, kv_weight, eps),
        apply_rope(k_pe, positions, base=rope_base),
    )


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    scores = torch.einsum("thd,shd->hts", q.float(), k.float()) * scale
    tokens = q.shape[0]
    mask = torch.ones(tokens, tokens, device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask.unsqueeze(0), -float("inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.einsum("hts,shd->thd", probs, v)


def swiglu(gate: torch.Tensor, up: torch.Tensor, *, clamp: float | None = None) -> torch.Tensor:
    if clamp is not None:
        gate = gate.clamp(min=-clamp, max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    return F.silu(gate) * up


def hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    shape = hidden_states.shape
    dtype = hidden_states.dtype
    x = hidden_states.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x, hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * hidden_states.float().view(shape), dim=1)
    return y.to(dtype)


def hc_post(
    update: torch.Tensor,
    residual: torch.Tensor,
    residual_mix: torch.Tensor,
    stream_scale: torch.Tensor,
) -> torch.Tensor:
    del residual_mix
    scale = stream_scale.view(1, -1, 1).to(update.dtype)
    return residual + update.unsqueeze(1) * scale


# ---------------------------------------------------------------------------
# Kernel dispatch
# ---------------------------------------------------------------------------


class KernelDispatcher:
    BACKENDS = ("triton", "cutedsl", "helion")

    def __init__(self, kernel_root: str | Path | None = None):
        self.kernel_root = Path(kernel_root) if kernel_root is not None else None
        self._modules: dict[tuple[str, str], ModuleType | None] = {}
        self._backend_modules_cache: dict[str, list[ModuleType]] = {}
        self._best: dict[tuple[Any, ...], str] = {}
        self._best_ops: dict[str, set[str]] = {}
        self._best_fast_path: dict[str, str] = {}
        self._load_best_config()

    @staticmethod
    def selected_backend(default: str = "torch") -> str:
        raw = os.getenv("RACETRACK_KERNEL_BACKEND", default).strip().lower()
        if raw == "cutedl":
            return "cutedsl"
        return raw

    def backend_status(self, backend: str) -> str:
        if backend == "torch":
            return "native"
        modules = self._load_backend_modules(backend)
        if not modules:
            return "missing"
        if any(bool(getattr(m, "BACKEND_AVAILABLE", False)) for m in modules):
            return "native"
        return "missing"

    def call(
        self,
        op_name: str,
        fallback: Fallback,
        *args: Any,
        backend: str | None = None,
        **kwargs: Any,
    ) -> Any:
        selected = backend or self.selected_backend(default="torch")
        if selected == "all":
            selected = "torch"
        if selected == "torch":
            return fallback(*args, **kwargs)
        if selected == "best":
            selected = self._best_fast_path.get(op_name)
            if selected is not None and selected != "torch" and self._resolve(selected, op_name) is None:
                selected = None
            if selected is None:
                selected = self._select_best(op_name, fallback, *args, **kwargs)
                self._best_fast_path[op_name] = selected
                self._save_best_config()
            self._best_ops.setdefault(op_name, set()).add(selected)
        if selected == "torch":
            return fallback(*args, **kwargs)
        fn = self._resolve(selected, op_name)
        if fn is None:
            self._handle_missing(selected, op_name)
        return fn(*args, fallback=fallback, **kwargs)

    def best_summary(self) -> str:
        if not self._best_ops:
            return "best"
        unique_backends = sorted(
            {backend for backends in self._best_ops.values() for backend in backends}
        )
        if len(unique_backends) == 1:
            return f"mixed={unique_backends[0]}"
        parts = [
            f"{op_name}={'+'.join(sorted(backends))}"
            for op_name, backends in sorted(self._best_ops.items())
        ]
        return "mixed=" + ";".join(parts)

    def _handle_missing(self, backend: str, op_name: str) -> None:
        raise RuntimeError(f"No available {backend} kernel found for {op_name}")

    def _resolve(self, backend: str, op_name: str) -> Callable[..., Any] | None:
        for module in self._load_backend_modules(backend):
            if not bool(getattr(module, "BACKEND_AVAILABLE", False)):
                continue
            fn = getattr(module, op_name, None)
            if callable(fn):
                return fn
        return None

    def _load_backend_modules(self, backend: str) -> list[ModuleType]:
        if backend in self._backend_modules_cache:
            return self._backend_modules_cache[backend]
        if self.kernel_root is None:
            return []
        backend_dir = self.kernel_root / backend
        if not backend_dir.is_dir():
            self._backend_modules_cache[backend] = []
            return []
        modules: list[ModuleType] = []
        for path in sorted(backend_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            module = self._load_module(backend, path.stem)
            if module is not None:
                modules.append(module)
        self._backend_modules_cache[backend] = modules
        return modules

    def _load_module(self, backend: str, module_name: str) -> ModuleType | None:
        if self.kernel_root is None:
            return None
        key = (backend, module_name)
        if key in self._modules:
            return self._modules[key]
        path = self.kernel_root / backend / f"{module_name}.py"
        if not path.exists():
            self._modules[key] = None
            return None
        spec_name = f"racetrack_partition_kernel_{abs(hash(path))}_{backend}_{module_name}"
        spec = importlib.util.spec_from_file_location(spec_name, path)
        if spec is None or spec.loader is None:
            self._modules[key] = None
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._modules[key] = module
        return module

    def _select_best(
        self,
        op_name: str,
        fallback: Fallback,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        key = self._best_key(op_name, args, kwargs)
        if key in self._best:
            selected = self._best[key]
            self._best_ops.setdefault(op_name, set()).add(selected)
            return selected

        timings: list[tuple[float, str]] = []
        _fb = fallback
        torch_elapsed = self._time_candidate(
            "torch", lambda *a, fallback=None, **kw: _fb(*a, **kw),
            fallback, *args, **kwargs,
        )
        timings.append((torch_elapsed, "torch"))
        for candidate in self.BACKENDS:
            fn = self._resolve(candidate, op_name)
            if fn is None:
                continue
            try:
                elapsed = self._time_candidate(candidate, fn, fallback, *args, **kwargs)
            except Exception:
                if os.getenv("RACETRACK_KERNEL_STRICT", "0") == "1":
                    raise
                continue
            timings.append((elapsed, candidate))
        selected = min(timings)[1]
        self._best[key] = selected
        self._best_ops.setdefault(op_name, set()).add(selected)
        return selected

    @staticmethod
    def _best_key(
        op_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, ...]:
        tensor_parts = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor_parts.append(
                    (
                        tuple(arg.shape),
                        str(arg.dtype),
                        str(arg.device),
                        tuple(arg.stride()),
                    )
                )
        scalar_parts = tuple(
            sorted(
                (key, value)
                for key, value in kwargs.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            )
        )
        return (op_name, tuple(tensor_parts), scalar_parts)

    @staticmethod
    def _time_candidate(
        backend: str,
        fn: Callable[..., Any],
        fallback: Fallback,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        def run_once() -> Any:
            return fn(*args, fallback=fallback, **kwargs)

        run_once()
        iterations = int(os.getenv("RACETRACK_BEST_ITERS", "10"))
        tensor_arg = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
        if tensor_arg is not None and tensor_arg.device.type == "cuda":
            torch.cuda.synchronize(tensor_arg.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                run_once()
            end.record()
            torch.cuda.synchronize(tensor_arg.device)
            return float(start.elapsed_time(end)) / iterations

        start_time = time.perf_counter()
        for _ in range(iterations):
            run_once()
        return (time.perf_counter() - start_time) * 1000.0 / iterations

    def _load_best_config(self) -> None:
        if self.kernel_root is None:
            return
        path = self.kernel_root / "best.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._best_fast_path.update(data)
        except (json.JSONDecodeError, OSError):
            pass

    def _save_best_config(self) -> None:
        if self.kernel_root is None:
            return
        path = self.kernel_root / "best.json"
        with open(path, "w") as f:
            json.dump(self._best_fast_path, f, indent=2, sort_keys=True)
            f.write("\n")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeepSeekConfig:
    name: str
    source: str
    vocab_size: int = 8192
    hidden_size: int = 512
    num_layers: int = 2
    num_attention_heads: int = 8
    head_dim: int = 64
    q_lora_rank: int = 128
    kv_lora_rank: int = 128
    qk_rope_head_dim: int = 16
    intermediate_size: int = 1024
    moe_intermediate_size: int = 384
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    n_shared_experts: int = 1
    rms_norm_eps: float = 1.0e-6
    rope_base: float = 10000.0
    routed_scaling_factor: float = 1.0
    hc_mult: int = 1
    hc_eps: float = 1.0e-5
    compress_ratio: int = 1
    seed: int = 1234

    def for_benchmark(self, **overrides: int | float | str) -> "DeepSeekConfig":
        return replace(self, **overrides)


DSV3_2_CONFIG = DeepSeekConfig(
    name="dsv3_2",
    source="vLLM PR 38595 specialized DeepSeek V3.2 NVFP4 path",
    hidden_size=512,
    num_layers=2,
    num_attention_heads=8,
    head_dim=64,
    q_lora_rank=128,
    kv_lora_rank=128,
    qk_rope_head_dim=16,
    moe_intermediate_size=384,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    seed=38595,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class RoutedMoE(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(config.hidden_size, config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
        self.shared = (
            SwiGLUExpert(config.hidden_size, shared_intermediate)
            if config.n_shared_experts
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids
        org_shape = hidden_states.shape
        x = hidden_states.reshape(-1, org_shape[-1])
        router_logits = F.linear(x.float(), self.gate.weight.float())
        topk_logits, topk_ids = torch.topk(
            router_logits, self.config.num_experts_per_tok, dim=-1
        )
        topk_weights = torch.softmax(topk_logits, dim=-1).to(x.dtype)

        out = torch.zeros_like(x)
        for slot in range(self.config.num_experts_per_tok):
            slot_ids = topk_ids[:, slot]
            slot_weights = topk_weights[:, slot].unsqueeze(-1)
            for expert_id, expert in enumerate(self.experts):
                mask = slot_ids == expert_id
                if bool(mask.any()):
                    out[mask] += expert(x[mask]) * slot_weights[mask]

        if self.shared is not None:
            out = out + self.shared(x)
        return (out * self.config.routed_scaling_factor).view(org_shape)


class FlattenedMLAAttention(nn.Module):
    def __init__(
        self,
        config: DeepSeekConfig,
        dispatcher: KernelDispatcher | None = None,
    ):
        super().__init__()
        if config.hidden_size != config.num_attention_heads * config.head_dim:
            raise ValueError(
                "This flattened baseline expects hidden_size == heads * head_dim."
            )
        if config.qk_rope_head_dim >= config.head_dim:
            raise ValueError("qk_rope_head_dim must be smaller than head_dim.")
        self.config = config
        self.dispatcher = dispatcher
        self.nope_dim = config.head_dim - config.qk_rope_head_dim

        qkv_out = config.q_lora_rank + config.kv_lora_rank + config.qk_rope_head_dim
        self.fused_qkv_a_proj = nn.Linear(config.hidden_size, qkv_out, bias=False)
        self.q_a_layernorm_weight = nn.Parameter(torch.ones(config.q_lora_rank))
        self.kv_a_layernorm_weight = nn.Parameter(torch.ones(config.kv_lora_rank))
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.num_attention_heads * (self.nope_dim + config.head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        config = self.config
        qkv = self.fused_qkv_a_proj(hidden_states)
        q_c, kv_c, k_pe = qkv.split(
            [config.q_lora_rank, config.kv_lora_rank, config.qk_rope_head_dim],
            dim=-1,
        )

        if self.dispatcher is None:
            q_c, kv_c, k_pe = fused_norm_rope(
                q_c,
                self.q_a_layernorm_weight,
                kv_c,
                self.kv_a_layernorm_weight,
                k_pe,
                positions,
                eps=config.rms_norm_eps,
                rope_base=config.rope_base,
            )
        else:
            q_c, kv_c, k_pe = self.dispatcher.call(
                "fused_norm_rope",
                fused_norm_rope,
                q_c,
                self.q_a_layernorm_weight,
                kv_c,
                self.kv_a_layernorm_weight,
                k_pe,
                positions,
                eps=config.rms_norm_eps,
                rope_base=config.rope_base,
            )

        tokens = hidden_states.shape[0]
        heads = config.num_attention_heads
        q = self.q_b_proj(q_c).view(tokens, heads, config.head_dim)
        q_nope, q_pe = q.split([self.nope_dim, config.qk_rope_head_dim], dim=-1)
        q_pe = apply_rope(
            q_pe,
            positions,
            rotary_dim=config.qk_rope_head_dim,
            base=config.rope_base,
        )
        q = torch.cat((q_nope, q_pe), dim=-1)

        kv = self.kv_b_proj(kv_c).view(tokens, heads, self.nope_dim + config.head_dim)
        k_nope, v = kv.split([self.nope_dim, config.head_dim], dim=-1)
        k = torch.cat((k_nope, k_pe.unsqueeze(1).expand(-1, heads, -1)), dim=-1)
        attn_out = causal_attention(q, k, v)
        return self.o_proj(attn_out.reshape(tokens, heads * config.head_dim))


class FlattenedDeepSeekBlock(nn.Module):
    def __init__(
        self,
        config: DeepSeekConfig,
        dispatcher: KernelDispatcher | None = None,
    ):
        super().__init__()
        self.config = config
        self.dispatcher = dispatcher
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = FlattenedMLAAttention(config, dispatcher)
        self.ffn = RoutedMoE(config)

        if config.hc_mult > 1:
            mix_dim = config.hc_mult * config.hidden_size
            self.hc_attn_fn = nn.Parameter(torch.empty(config.hc_mult, mix_dim))
            self.hc_attn_scale = nn.Parameter(torch.ones(config.hc_mult))
            self.hc_attn_base = nn.Parameter(torch.zeros(config.hc_mult))
            self.hc_ffn_fn = nn.Parameter(torch.empty(config.hc_mult, mix_dim))
            self.hc_ffn_scale = nn.Parameter(torch.ones(config.hc_mult))
            self.hc_ffn_base = nn.Parameter(torch.zeros(config.hc_mult))
            self.attn_stream_scale = nn.Parameter(torch.linspace(1.0, 0.5, config.hc_mult))
            self.ffn_stream_scale = nn.Parameter(torch.linspace(0.5, 1.0, config.hc_mult))
        else:
            self.register_parameter("hc_attn_fn", None)
            self.register_parameter("hc_attn_scale", None)
            self.register_parameter("hc_attn_base", None)
            self.register_parameter("hc_ffn_fn", None)
            self.register_parameter("hc_ffn_scale", None)
            self.register_parameter("hc_ffn_base", None)
            self.register_parameter("attn_stream_scale", None)
            self.register_parameter("ffn_stream_scale", None)

    def reset_hc_parameters(self) -> None:
        if self.config.hc_mult <= 1:
            return
        nn.init.normal_(self.hc_attn_fn, std=0.02)
        nn.init.normal_(self.hc_ffn_fn, std=0.02)

    def _hc_head(
        self,
        hidden_states: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        if self.dispatcher is None:
            return hc_head(
                hidden_states,
                fn,
                scale,
                base,
                rms_norm_eps=self.config.rms_norm_eps,
                hc_eps=self.config.hc_eps,
            )
        return self.dispatcher.call(
            "hc_head",
            hc_head,
            hidden_states,
            fn,
            scale,
            base,
            rms_norm_eps=self.config.rms_norm_eps,
            hc_eps=self.config.hc_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.config.hc_mult <= 1:
            residual = hidden_states
            x = self.attn_norm(hidden_states)
            hidden_states = residual + self.attn(x, positions)
            residual = hidden_states
            x = self.ffn_norm(hidden_states)
            hidden_states = residual + self.ffn(x, input_ids)
            return hidden_states

        residual = hidden_states
        x = self._hc_head(hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, positions)
        hidden_states = hc_post(x, residual, x, self.attn_stream_scale)

        residual = hidden_states
        x = self._hc_head(hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        return hc_post(x, residual, x, self.ffn_stream_scale)


class FlattenedDeepSeekModel(nn.Module):
    def __init__(
        self,
        config: DeepSeekConfig,
        *,
        partition_root: str | Path | None = None,
    ):
        super().__init__()
        self.config = config
        self.dispatcher = (
            KernelDispatcher(Path(partition_root) / "kernels")
            if partition_root is not None
            else None
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = nn.ModuleList(
                [FlattenedDeepSeekBlock(config, self.dispatcher) for _ in range(config.num_layers)]
            )
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            if config.hc_mult > 1:
                mix_dim = config.hc_mult * config.hidden_size
                self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, mix_dim))
                self.hc_head_scale = nn.Parameter(torch.ones(config.hc_mult))
                self.hc_head_base = nn.Parameter(torch.zeros(config.hc_mult))
            else:
                self.register_parameter("hc_head_fn", None)
                self.register_parameter("hc_head_scale", None)
                self.register_parameter("hc_head_base", None)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self._reset_extra_parameters()

    def _reset_extra_parameters(self) -> None:
        if self.config.hc_mult > 1:
            nn.init.normal_(self.hc_head_fn, std=0.02)
        for layer in self.layers:
            if isinstance(layer, FlattenedDeepSeekBlock):
                layer.reset_hc_parameters()

    @property
    def backend_status(self) -> dict[str, str]:
        if self.dispatcher is None:
            return {"torch": "native"}
        return {
            backend: self.dispatcher.backend_status(backend)
            for backend in ("triton", "cutedsl", "helion")
        } | {"torch": "native"}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat_input_ids = input_ids.reshape(-1)
        if positions is None:
            positions = torch.arange(
                flat_input_ids.numel(),
                device=flat_input_ids.device,
                dtype=torch.long,
            )
        else:
            positions = positions.reshape(-1).to(device=flat_input_ids.device, dtype=torch.long)

        hidden_states = self.embed_tokens(flat_input_ids)
        if self.config.hc_mult > 1:
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.config.hc_mult, 1)

        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, flat_input_ids)

        if self.config.hc_mult > 1:
            if self.dispatcher is None:
                hidden_states = hc_head(
                    hidden_states,
                    self.hc_head_fn,
                    self.hc_head_scale,
                    self.hc_head_base,
                    rms_norm_eps=self.config.rms_norm_eps,
                    hc_eps=self.config.hc_eps,
                )
            else:
                hidden_states = self.dispatcher.call(
                    "hc_head",
                    hc_head,
                    hidden_states,
                    self.hc_head_fn,
                    self.hc_head_scale,
                    self.hc_head_base,
                    rms_norm_eps=self.config.rms_norm_eps,
                    hc_eps=self.config.hc_eps,
                )

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


def build_model(**overrides) -> FlattenedDeepSeekModel:
    config = DSV3_2_CONFIG.for_benchmark(**overrides)
    return FlattenedDeepSeekModel(config, partition_root=Path(__file__).parent)
