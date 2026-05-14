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

MODEL_NAME = "dsv3_2_nvfp4"
PARTITION_NOTES = (
    "4-kernel fusion matching the NVFP4 reference diagram. "
    "Kernel 1: AR+Add+RMS + QKV A Proj + Indexer K. "
    "Kernel 2: LayerNorm + RoPE + Quantize FP8 + Indexer Cache. "
    "Kernel 3: Q/Indexer projections + scoring + W_UK_T. "
    "Kernel 4: Q RoPE + Cat + Q Quantize FP8."
)

FUSED_OP_GRAPH = {
    "fused_ar_rms_qkv_proj": ["ar_add_rms", "qkv_a_proj", "indexer_k_proj"],
    "fused_indexer_k_path": ["indexer_ln", "indexer_rope", "indexer_quant_fp8", "indexer_cache"],
    "fused_q_indexer_score": [
        "q_rms", "q_b_proj", "indexer_w", "indexer_q_proj",
        "indexer_q_rope", "indexer_q_fp8", "w_uk_t",
        "indexer_w_scale", "indexer_mqa",
    ],
    "fused_q_rope_quant": ["q_rope", "cat_q", "q_quant_fp8"],
}

Fallback = Callable[..., Any]


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x_float = x.float()
    scale = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    return (x_float * scale * weight.float()).to(orig_dtype)


def rms_norm_with_residual(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = x.dtype
    if residual is not None:
        hidden = x.float() + residual.float()
    else:
        hidden = x.float()
    var = hidden.square().mean(dim=-1, keepdim=True)
    normed = hidden * torch.rsqrt(var + eps)
    return (weight.float() * normed).to(dtype), hidden.to(dtype)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dim: int,
    eps: float,
) -> torch.Tensor:
    return F.layer_norm(x.float(), (dim,), weight, bias, eps).type_as(x)


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    *,
    base: float = 10000.0,
    factor: float = 1.0,
    original_seq_len: int = 4096,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if max_seq_len > original_seq_len and factor > 1.0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        linear_func = (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 0.001)
        smooth = torch.clamp(linear_func, 0, 1)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    interleaved: bool = True,
) -> torch.Tensor:
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x_c = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x_c.size(1), 1, x_c.size(-1))
    y = torch.view_as_real(x_c * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)


def hadamard_transform(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return (x.float() @ H[:d, :d].float() * (d ** -0.5)).type_as(x)


def build_hadamard_matrix(dim: int, device: torch.device) -> torch.Tensor:
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < dim:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H


_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0


def act_quant(
    x: torch.Tensor, block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    N = x.size(-1)
    if N % block_size != 0:
        try:
            return x.to(_FP8_DTYPE), torch.ones(
                *x.shape[:-1], 1, dtype=torch.float32, device=x.device,
            )
        except RuntimeError:
            return x.float(), torch.ones(
                *x.shape[:-1], 1, dtype=torch.float32, device=x.device,
            )
    n_groups = N // block_size
    shape = x.shape
    x_flat = x.float().reshape(-1, n_groups, block_size)
    amax = x_flat.abs().amax(dim=-1).clamp(min=1e-4)
    scale = amax / _FP8_MAX
    x_scaled = x_flat / scale.unsqueeze(-1)
    try:
        y = x_scaled.clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE).view(shape)
    except RuntimeError:
        y = x_scaled.clamp(-_FP8_MAX, _FP8_MAX).float().view(shape)
    s = scale.view(*shape[:-1], n_groups)
    return y, s


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    b, m, h, d = q.shape
    logits = torch.einsum("bmhd,bnd->bmhn", q.float(), k.float())
    logits = torch.relu(logits) * q_s.unsqueeze(-1)
    logits_sum = logits.sum(dim=2)
    return logits_sum * k_s.unsqueeze(1)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate.float()).to(gate.dtype) * up


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    scores = torch.einsum("bshd,bthd->bsht", q.float(), k.float()) * scale
    if mask is not None:
        scores = scores + mask.unsqueeze(2)
    scores = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.einsum("bsht,bthd->bshd", scores, v)


# ---------------------------------------------------------------------------
# Fused ops (partition dispatch targets)
# ---------------------------------------------------------------------------


def fused_ar_rms_qkv_proj(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    norm_weight: torch.Tensor,
    wq_a_weight: torch.Tensor,
    wkv_a_weight: torch.Tensor,
    indexer_wk_weight: torch.Tensor,
    *,
    eps: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if residual is not None:
        hidden = x.float() + residual.float()
    else:
        hidden = x.float()
    dtype = x.dtype
    var = hidden.square().mean(dim=-1, keepdim=True)
    normed = (norm_weight.float() * hidden * torch.rsqrt(var + eps)).to(dtype)
    residual_out = hidden.to(dtype)
    qkv = F.linear(normed, torch.cat([wq_a_weight, wkv_a_weight], dim=0))
    total = q_lora_rank + kv_lora_rank + qk_rope_head_dim
    q_c = qkv[..., :q_lora_rank]
    kv_c = qkv[..., q_lora_rank:q_lora_rank + kv_lora_rank]
    k_pe = qkv[..., q_lora_rank + kv_lora_rank:total]
    indexer_k = F.linear(normed, indexer_wk_weight)
    return residual_out, normed, q_c, kv_c, k_pe, indexer_k


def fused_indexer_k_path(
    k: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    freqs_cis: torch.Tensor,
    H: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    *,
    ln_dim: int,
    ln_eps: float,
    rope_head_dim: int,
    start_pos: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = layer_norm(k, ln_weight, ln_bias, ln_dim, ln_eps)
    k_pe, k_nope = torch.split(k, [rope_head_dim, k.shape[-1] - rope_head_dim], dim=-1)
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)
    k = hadamard_transform(k, H)
    k_fp8, k_scale = act_quant(k, block_size)
    bsz, seqlen = k_fp8.shape[0], k_fp8.shape[1]
    end_pos = start_pos + seqlen
    k_cache[:bsz, start_pos:end_pos] = k_fp8.float()
    k_scale_cache[:bsz, start_pos:end_pos] = k_scale
    return k_fp8, k_scale


def fused_q_indexer_score(
    qr: torch.Tensor,
    normed_x: torch.Tensor,
    wq_b_weight: torch.Tensor,
    idx_wq_b_weight: torch.Tensor,
    idx_weights_proj_weight: torch.Tensor,
    wkv_b_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    H: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    *,
    eps: float,
    n_heads: int,
    qk_head_dim: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    idx_n_heads: int,
    idx_head_dim: int,
    rope_head_dim: int,
    softmax_scale: float,
    idx_softmax_scale: float,
    start_pos: int,
    end_pos: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seqlen, _ = qr.shape

    # MLA Q B projection
    q = F.linear(qr, wq_b_weight).view(bsz, seqlen, n_heads, qk_head_dim)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_head_dim - qk_nope_head_dim], dim=-1)

    # Absorbed W_UK_T for MLA decode
    wkv_b = wkv_b_weight.view(n_heads, -1, kv_lora_rank)
    q_nope_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope_head_dim])

    # Indexer Q path
    idx_q = F.linear(qr, idx_wq_b_weight).view(bsz, seqlen, idx_n_heads, idx_head_dim)
    idx_q_pe, idx_q_nope = torch.split(idx_q, [rope_head_dim, idx_head_dim - rope_head_dim], dim=-1)
    idx_q_pe = apply_rotary_emb(idx_q_pe, freqs_cis, interleaved=False)
    idx_q = torch.cat([idx_q_pe, idx_q_nope], dim=-1)
    idx_q = hadamard_transform(idx_q, H)
    idx_q_fp8, idx_q_scale = act_quant(idx_q, block_size)

    # Indexer W + scale
    weights = F.linear(normed_x.float(), idx_weights_proj_weight.float()) * idx_n_heads ** -0.5
    weights = (weights.unsqueeze(-1) * idx_q_scale * idx_softmax_scale).squeeze(-1)

    # Indexer MQA scoring
    k_s = k_scale_cache[:bsz, :end_pos].squeeze(-1).contiguous()
    k_cached = k_cache[:bsz, :end_pos].contiguous()
    index_score = fp8_index(idx_q_fp8.float(), weights, k_cached, k_s)
    topk_indices = index_score.topk(min(end_pos, seqlen * 2), dim=-1)[1]

    return q_nope, q_nope_absorbed, q_pe, topk_indices


def fused_q_rope_quant(
    q_pe: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=True)
    return q_pe


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
        self, op_name: str, fallback: Fallback, *args: Any, **kwargs: Any,
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
    def _best_key(op_name, args, kwargs):
        tensor_parts = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor_parts.append((tuple(arg.shape), str(arg.dtype), str(arg.device), tuple(arg.stride())))
        scalar_parts = tuple(sorted(
            (k, v) for k, v in kwargs.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        ))
        return (op_name, tuple(tensor_parts), scalar_parts)

    @staticmethod
    def _time_candidate(backend, fn, fallback, *args, **kwargs):
        def run_once():
            return fn(*args, fallback=fallback, **kwargs)
        run_once()
        iterations = int(os.getenv("RACETRACK_BEST_ITERS", "10"))
        tensor_arg = next((a for a in args if isinstance(a, torch.Tensor)), None)
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

    def _load_best_config(self):
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

    def _save_best_config(self):
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
class DeepSeekNVFP4Config:
    name: str
    source: str
    vocab_size: int = 8192
    hidden_size: int = 512
    num_layers: int = 2
    num_attention_heads: int = 8
    head_dim: int = 64
    q_lora_rank: int = 128
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 48
    qk_rope_head_dim: int = 16
    v_head_dim: int = 64
    intermediate_size: int = 1024
    moe_intermediate_size: int = 384
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    n_shared_experts: int = 1
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: str = "softmax"
    route_scale: float = 1.0
    rms_norm_eps: float = 1.0e-6
    rope_base: float = 10000.0
    rope_factor: float = 1.0
    original_seq_len: int = 4096
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    index_n_heads: int = 8
    index_head_dim: int = 64
    index_topk: int = 128
    max_batch_size: int = 1
    max_seq_len: int = 4096
    block_size: int = 128
    seed: int = 1234

    def for_benchmark(self, **overrides: int | float | str) -> "DeepSeekNVFP4Config":
        return replace(self, **overrides)


DSV3_2_NVFP4_CONFIG = DeepSeekNVFP4Config(
    name="dsv3_2_nvfp4",
    source="DeepSeek V3.2 NVFP4 decode path with Indexer + FP8 + KV cache",
    hidden_size=512,
    num_layers=2,
    num_attention_heads=8,
    head_dim=64,
    q_lora_rank=128,
    kv_lora_rank=128,
    qk_nope_head_dim=48,
    qk_rope_head_dim=16,
    v_head_dim=64,
    moe_intermediate_size=384,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    index_n_heads=8,
    index_head_dim=64,
    index_topk=128,
    max_batch_size=1,
    max_seq_len=4096,
    seed=38595,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = eps

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return rms_norm(x, self.weight, self.eps)
        return rms_norm_with_residual(x, residual, self.weight, self.eps)


class LayerNormModule(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm(x, self.weight, self.bias, self.dim, self.eps)


class Indexer(nn.Module):
    def __init__(self, config: DeepSeekNVFP4Config):
        super().__init__()
        self.config = config
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.block_size = config.block_size
        self.softmax_scale = self.head_dim ** -0.5

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = LayerNormModule(self.head_dim)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)

        self.register_buffer(
            "hadamard_matrix",
            build_hadamard_matrix(self.head_dim, torch.device("cpu")),
            persistent=False,
        )
        self.register_buffer(
            "k_cache",
            torch.zeros(config.max_batch_size, config.max_seq_len, self.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "k_scale_cache",
            torch.zeros(
                config.max_batch_size, config.max_seq_len,
                max(self.head_dim // config.block_size, 1),
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1,
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=False)
        q = torch.cat([q_pe, q_nope], dim=-1)

        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, k.shape[-1] - self.rope_head_dim], dim=-1,
        )
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        q = hadamard_transform(q, self.hadamard_matrix)
        k = hadamard_transform(k, self.hadamard_matrix)

        q_fp8, q_scale = act_quant(q, self.block_size)
        k_fp8, k_scale = act_quant(k, self.block_size)

        self.k_cache[:bsz, start_pos:end_pos] = k_fp8.float()
        self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale

        weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
        weights = (weights.unsqueeze(-1) * q_scale * self.softmax_scale).squeeze(-1)

        k_s = self.k_scale_cache[:bsz, :end_pos].squeeze(-1).contiguous()
        k_cached = self.k_cache[:bsz, :end_pos].contiguous()
        index_score = fp8_index(q_fp8.float(), weights, k_cached, k_s)

        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        return topk_indices


class FlattenedMLAAttention(nn.Module):
    def __init__(self, config: DeepSeekNVFP4Config, dispatcher: KernelDispatcher | None = None):
        super().__init__()
        self.config = config
        self.dispatcher = dispatcher
        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.softmax_scale = self.qk_head_dim ** -0.5

        if config.max_seq_len > config.original_seq_len and config.mscale != 1.0:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.wq_a = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
        self.wkv_a = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, config.hidden_size, bias=False)

        self.indexer = Indexer(config)

        self.register_buffer(
            "kv_cache",
            torch.zeros(config.max_batch_size, config.max_seq_len, self.kv_lora_rank),
            persistent=False,
        )
        self.register_buffer(
            "pe_cache",
            torch.zeros(config.max_batch_size, config.max_seq_len, self.qk_rope_head_dim),
            persistent=False,
        )

    def forward(
        self,
        normed_x: torch.Tensor,
        q_c: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        indexer_k: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = normed_x.size()
        end_pos = start_pos + seqlen
        config = self.config

        qr = self.q_norm(q_c)

        # Kernel 2: Indexer K path (LayerNorm + RoPE + FP8 quant + cache)
        if self.dispatcher is None:
            fused_indexer_k_path(
                indexer_k,
                self.indexer.k_norm.weight, self.indexer.k_norm.bias,
                freqs_cis, self.indexer.hadamard_matrix,
                self.indexer.k_cache, self.indexer.k_scale_cache,
                ln_dim=self.indexer.head_dim, ln_eps=self.indexer.k_norm.eps,
                rope_head_dim=config.qk_rope_head_dim,
                start_pos=start_pos, block_size=config.block_size,
            )
        else:
            self.dispatcher.call(
                "fused_indexer_k_path", fused_indexer_k_path,
                indexer_k,
                self.indexer.k_norm.weight, self.indexer.k_norm.bias,
                freqs_cis, self.indexer.hadamard_matrix,
                self.indexer.k_cache, self.indexer.k_scale_cache,
                ln_dim=self.indexer.head_dim, ln_eps=self.indexer.k_norm.eps,
                rope_head_dim=config.qk_rope_head_dim,
                start_pos=start_pos, block_size=config.block_size,
            )

        # Kernel 3: Q/Indexer scoring + W_UK_T
        if self.dispatcher is None:
            q_nope, q_nope_absorbed, q_pe, topk_indices = fused_q_indexer_score(
                qr, normed_x,
                self.wq_b.weight, self.indexer.wq_b.weight,
                self.indexer.weights_proj.weight,
                self.wkv_b.weight,
                freqs_cis, self.indexer.hadamard_matrix,
                self.indexer.k_cache, self.indexer.k_scale_cache,
                eps=config.rms_norm_eps,
                n_heads=self.n_heads, qk_head_dim=self.qk_head_dim,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                idx_n_heads=self.indexer.n_heads,
                idx_head_dim=self.indexer.head_dim,
                rope_head_dim=config.qk_rope_head_dim,
                softmax_scale=self.softmax_scale,
                idx_softmax_scale=self.indexer.softmax_scale,
                start_pos=start_pos, end_pos=end_pos,
                block_size=config.block_size,
            )
        else:
            q_nope, q_nope_absorbed, q_pe, topk_indices = self.dispatcher.call(
                "fused_q_indexer_score", fused_q_indexer_score,
                qr, normed_x,
                self.wq_b.weight, self.indexer.wq_b.weight,
                self.indexer.weights_proj.weight,
                self.wkv_b.weight,
                freqs_cis, self.indexer.hadamard_matrix,
                self.indexer.k_cache, self.indexer.k_scale_cache,
                eps=config.rms_norm_eps,
                n_heads=self.n_heads, qk_head_dim=self.qk_head_dim,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                idx_n_heads=self.indexer.n_heads,
                idx_head_dim=self.indexer.head_dim,
                rope_head_dim=config.qk_rope_head_dim,
                softmax_scale=self.softmax_scale,
                idx_softmax_scale=self.indexer.softmax_scale,
                start_pos=start_pos, end_pos=end_pos,
                block_size=config.block_size,
            )

        # Kernel 4: Q RoPE
        if self.dispatcher is None:
            q_pe = fused_q_rope_quant(q_pe, freqs_cis, block_size=config.block_size)
        else:
            q_pe = self.dispatcher.call(
                "fused_q_rope_quant", fused_q_rope_quant,
                q_pe, freqs_cis, block_size=config.block_size,
            )

        # KV norm + RoPE + cache (not fused — these are simple ops)
        kv_c = self.kv_norm(kv_c)
        k_pe_roped = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True)
        self.kv_cache[:bsz, start_pos:end_pos] = kv_c
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe_roped.squeeze(2)

        if mask is not None:
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv_expanded = self.wkv_b(kv_c)
            kv_expanded = kv_expanded.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe_roped.expand(-1, -1, self.n_heads, -1)], dim=-1)

            scores = torch.einsum("bshd,bthd->bsht", q.float(), k.float()) * self.softmax_scale
            index_mask = torch.full(
                (bsz, seqlen, seqlen), float("-inf"), device=normed_x.device,
            ).scatter_(-1, topk_indices, 0)
            scores = scores + (index_mask + mask).unsqueeze(2)
            scores = scores.softmax(dim=-1).to(v.dtype)
            x = torch.einsum("bsht,bthd->bshd", scores, v)
        else:
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope_absorbed, self.kv_cache[:bsz, :end_pos])
                + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            ) * self.softmax_scale

            index_mask = torch.full(
                (bsz, 1, end_pos), float("-inf"), device=normed_x.device,
            ).scatter_(-1, topk_indices, 0)
            scores = scores + index_mask.unsqueeze(2)
            scores = scores.softmax(dim=-1)

            wkv_b = self.wkv_b.weight.view(self.n_heads, -1, self.kv_lora_rank)
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])

        x = self.wo(x.flatten(2))
        return x


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class Gate(nn.Module):
    def __init__(self, config: DeepSeekNVFP4Config):
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            group_scores = scores.amax(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class RoutedMoE(nn.Module):
    def __init__(self, config: DeepSeekNVFP4Config):
        super().__init__()
        self.config = config
        self.gate = Gate(config)
        self.experts = nn.ModuleList([
            SwiGLUExpert(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.n_routed_experts)
        ])
        shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
        self.shared = (
            SwiGLUExpert(config.hidden_size, shared_intermediate)
            if config.n_shared_experts else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.config.hidden_size)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        for i, expert in enumerate(self.experts):
            idx, top = torch.where(indices == i)
            if idx.numel() == 0:
                continue
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        if self.shared is not None:
            y = y + self.shared(x)
        return y.type_as(x).view(shape)


class FlattenedDeepSeekBlock(nn.Module):
    def __init__(self, config: DeepSeekNVFP4Config, dispatcher: KernelDispatcher | None = None):
        super().__init__()
        self.config = config
        self.dispatcher = dispatcher
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = FlattenedMLAAttention(config, dispatcher)
        self.ffn = RoutedMoE(config)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config

        # Kernel 1: AR + Add + RMS + QKV A Proj + Indexer K
        if self.dispatcher is None:
            residual_out, normed, q_c, kv_c, k_pe, indexer_k = fused_ar_rms_qkv_proj(
                x, residual, self.attn_norm.weight,
                self.attn.wq_a.weight, self.attn.wkv_a.weight,
                self.attn.indexer.wk.weight,
                eps=config.rms_norm_eps,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_rope_head_dim=config.qk_rope_head_dim,
            )
        else:
            residual_out, normed, q_c, kv_c, k_pe, indexer_k = self.dispatcher.call(
                "fused_ar_rms_qkv_proj", fused_ar_rms_qkv_proj,
                x, residual, self.attn_norm.weight,
                self.attn.wq_a.weight, self.attn.wkv_a.weight,
                self.attn.indexer.wk.weight,
                eps=config.rms_norm_eps,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_rope_head_dim=config.qk_rope_head_dim,
            )

        x = self.attn(normed, q_c, kv_c, k_pe, indexer_k, start_pos, freqs_cis, mask)
        x, residual = self.ffn_norm(x, residual_out)
        x = self.ffn(x)
        return x, residual


class FlattenedDeepSeekModel(nn.Module):
    def __init__(
        self,
        config: DeepSeekNVFP4Config,
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
            self.layers = nn.ModuleList([
                FlattenedDeepSeekBlock(config, self.dispatcher)
                for _ in range(config.num_layers)
            ])
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.qk_rope_head_dim,
                config.max_seq_len,
                base=config.rope_base,
                factor=config.rope_factor,
                original_seq_len=config.original_seq_len,
                beta_fast=config.beta_fast,
                beta_slow=config.beta_slow,
            ),
            persistent=False,
        )

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
        start_pos: int = 0,
    ) -> torch.Tensor:
        flat_input_ids = input_ids.reshape(-1)
        seqlen = flat_input_ids.numel()
        if positions is None:
            positions = torch.arange(
                start_pos, start_pos + seqlen,
                device=flat_input_ids.device, dtype=torch.long,
            )
        else:
            positions = positions.reshape(-1).to(device=flat_input_ids.device, dtype=torch.long)
            start_pos = positions[0].item()

        h = self.embed_tokens(flat_input_ids).unsqueeze(0)

        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen].to(h.device)
        mask = (
            torch.full((seqlen, seqlen), float("-inf"), device=h.device).triu_(1)
            if seqlen > 1
            else None
        )

        residual = None
        for layer in self.layers:
            h, residual = layer(h, residual, start_pos, freqs_cis, mask)

        h, _ = self.norm(h, residual)
        logits = self.lm_head(h.squeeze(0))
        return logits


def build_model(**overrides) -> FlattenedDeepSeekModel:
    config = DSV3_2_NVFP4_CONFIG.for_benchmark(**overrides)
    return FlattenedDeepSeekModel(config, partition_root=Path(__file__).parent)
