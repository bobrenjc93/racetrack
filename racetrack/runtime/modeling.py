from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .dispatch import KernelDispatcher
from . import torch_ops


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


def model_config(name: str) -> DeepSeekConfig:
    key = name.lower().replace("-", "_")
    if key in {"dsv3_2", "ds3_2", "deepseek_v3_2"}:
        return DeepSeekConfig(
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
    if key in {"dsv4", "deepseek_v4"}:
        return DeepSeekConfig(
            name="dsv4",
            source="vLLM PR 40860 DeepSeek V4 model path",
            hidden_size=512,
            num_layers=2,
            num_attention_heads=8,
            head_dim=64,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_rope_head_dim=16,
            moe_intermediate_size=512,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            hc_mult=2,
            compress_ratio=4,
            seed=40860,
        )
    if key in {"ds", "deepseek"}:
        return DeepSeekConfig(
            name="ds",
            source="Generic flattened DeepSeek MLA/MoE path seeded from PR 40860 shapes",
            hidden_size=384,
            num_layers=2,
            num_attention_heads=6,
            head_dim=64,
            q_lora_rank=96,
            kv_lora_rank=96,
            qk_rope_head_dim=16,
            moe_intermediate_size=256,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            seed=40861,
        )
    raise KeyError(f"Unknown model config: {name}")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_ops.rms_norm(x, self.weight, self.eps)


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch_ops.swiglu(self.w1(x), self.w3(x)))


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

        fallback = torch_ops.fused_norm_rope
        if self.dispatcher is None:
            q_c, kv_c, k_pe = fallback(
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
                fallback,
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
        q_pe = torch_ops.apply_rope(
            q_pe,
            positions,
            rotary_dim=config.qk_rope_head_dim,
            base=config.rope_base,
        )
        q = torch.cat((q_nope, q_pe), dim=-1)

        kv = self.kv_b_proj(kv_c).view(tokens, heads, self.nope_dim + config.head_dim)
        k_nope, v = kv.split([self.nope_dim, config.head_dim], dim=-1)
        k = torch.cat((k_nope, k_pe.unsqueeze(1).expand(-1, heads, -1)), dim=-1)
        attn_out = torch_ops.causal_attention(q, k, v)
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
        fallback = torch_ops.hc_head
        if self.dispatcher is None:
            return fallback(
                hidden_states,
                fn,
                scale,
                base,
                rms_norm_eps=self.config.rms_norm_eps,
                hc_eps=self.config.hc_eps,
            )
        return self.dispatcher.call(
            "hc_head",
            fallback,
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
        hidden_states = torch_ops.hc_post(x, residual, x, self.attn_stream_scale)

        residual = hidden_states
        x = self._hc_head(hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        return torch_ops.hc_post(x, residual, x, self.ffn_stream_scale)


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
            fallback = torch_ops.hc_head
            if self.dispatcher is None:
                hidden_states = fallback(
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
                    fallback,
                    hidden_states,
                    self.hc_head_fn,
                    self.hc_head_scale,
                    self.hc_head_base,
                    rms_norm_eps=self.config.rms_norm_eps,
                    hc_eps=self.config.hc_eps,
                )

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
