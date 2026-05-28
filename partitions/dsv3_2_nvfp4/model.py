from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from partitions._base import (
    Fallback,
    KernelDispatcher,
    SwiGLUExpert,
    rms_norm,
)

MODEL_NAME = "dsv3_2_nvfp4"


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------


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
    return F.layer_norm(x.float(), (dim,), weight.float(), bias.float(), eps).type_as(x)


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


def build_hadamard_matrix(dim: int) -> torch.Tensor:
    H = torch.tensor([[1.0]])
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
            build_hadamard_matrix(self.head_dim),
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

        weights = F.linear(x.float(), self.weights_proj.weight.float()) * self.n_heads ** -0.5
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
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=True)

        kv = self.wkv_a(x)
        kv_c, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c = self.kv_norm(kv_c)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True)

        self.kv_cache[:bsz, start_pos:end_pos] = kv_c
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

        topk_indices = self.indexer(x, qr, start_pos, freqs_cis)

        if mask is not None:
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv_expanded = self.wkv_b(kv_c)
            kv_expanded = kv_expanded.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

            scores = torch.einsum("bshd,bthd->bsht", q.float(), k.float()) * self.softmax_scale
            index_mask = torch.full(
                (bsz, seqlen, seqlen), float("-inf"), device=x.device,
            ).scatter_(-1, topk_indices, 0)
            scores = scores + (index_mask + mask).unsqueeze(2)
            scores = scores.softmax(dim=-1).to(v.dtype)
            x = torch.einsum("bsht,bthd->bshd", scores, v)
        else:
            wkv_b = self.wkv_b.weight.view(self.n_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
                + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            ) * self.softmax_scale

            index_mask = torch.full(
                (bsz, 1, end_pos), float("-inf"), device=x.device,
            ).scatter_(-1, topk_indices, 0)
            scores = scores + index_mask.unsqueeze(2)
            scores = scores.softmax(dim=-1)

            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])

        x = self.wo(x.flatten(2))
        return x


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
        for expert_id, expert in enumerate(self.experts):
            weight = torch.zeros(x.shape[0], 1, device=x.device, dtype=weights.dtype)
            for slot in range(self.config.num_experts_per_tok):
                selected = (indices[:, slot] == expert_id).unsqueeze(-1).to(weights.dtype)
                weight = weight + selected * weights[:, slot : slot + 1]
            y = y + expert(x) * weight
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
        if residual is None:
            x, residual = self.attn_norm(x), x
        else:
            x, residual = self.attn_norm(x, residual)
        x = self.attn(x, start_pos, freqs_cis, mask)
        x, residual = self.ffn_norm(x, residual)
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
        self.register_buffer(
            "_causal_mask",
            torch.full((config.max_seq_len, config.max_seq_len), float("-inf")).triu_(1),
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

        h = self.embed_tokens(flat_input_ids).unsqueeze(0)

        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen].to(h.device)
        mask = self._causal_mask[:seqlen, :seqlen] if seqlen > 1 else None

        residual = None
        for layer in self.layers:
            h, residual = layer(h, residual, start_pos, freqs_cis, mask)

        h, _ = self.norm(h, residual)
        logits = self.lm_head(h.squeeze(0))
        return logits


def build_model(**overrides) -> FlattenedDeepSeekModel:
    config = DSV3_2_NVFP4_CONFIG.for_benchmark(**overrides)
    return FlattenedDeepSeekModel(config)
