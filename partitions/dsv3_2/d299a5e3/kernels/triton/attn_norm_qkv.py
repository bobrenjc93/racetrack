from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    BACKEND_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    BACKEND_AVAILABLE = False


if BACKEND_AVAILABLE:

    @triton.jit
    def _rms_norm_kernel(
        x, weight, out,
        eps: tl.constexpr, cols: tl.constexpr,
        stride_t: tl.constexpr, block_size: tl.constexpr,
    ):
        token = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < cols
        values = tl.load(x + token * stride_t + offsets, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / cols
        scale = tl.rsqrt(variance + eps)
        tl.store(out + token * cols + offsets, values * scale * weights, mask=mask)


def _triton_rms_norm(x, weight, eps):
    x_c = x.contiguous()
    cols = x_c.shape[-1]
    block_size = triton.next_power_of_2(cols)
    out = torch.empty_like(x_c)
    _rms_norm_kernel[(x_c.shape[0],)](
        x_c, weight.contiguous(), out,
        eps, cols, x_c.stride(0), block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return out


def fused_attn_norm_qkv(
    hidden_states, norm_weight, qkv_weight,
    *, eps, q_lora_rank, kv_lora_rank, qk_rope_head_dim, fallback,
):
    del fallback
    x = _triton_rms_norm(hidden_states, norm_weight, eps)
    qkv = F.linear(x, qkv_weight)
    q_c, kv_c, k_pe = qkv.split(
        [q_lora_rank, kv_lora_rank, qk_rope_head_dim], dim=-1,
    )
    return q_c, kv_c, k_pe
