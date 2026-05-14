from __future__ import annotations

import math
import os

import torch

try:
    import helion
    import helion.language as hl

    BACKEND_AVAILABLE = True
except Exception:
    helion = None
    hl = None
    BACKEND_AVAILABLE = False


def _autotune_effort() -> str:
    return os.getenv("RACETRACK_HELION_AUTOTUNE_EFFORT", "quick")


if BACKEND_AVAILABLE:

    @helion.kernel(autotune_effort=_autotune_effort())
    def _rms_norm_kernel(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        out = torch.empty_like(x)
        tokens, _hidden = x.size()
        for tile_t in hl.tile(tokens):
            values = x[tile_t, :]
            values_f32 = values.to(torch.float32)
            variance = torch.mean(values_f32 * values_f32, dim=1)
            scale = torch.rsqrt(variance + eps).view(tile_t, 1)
            out[tile_t, :] = (
                values_f32 * scale * weight[:].to(torch.float32)
            ).to(x.dtype)
        return out


def fused_norms_rope_cache(
    q_c, q_norm_weight,
    kv_c, kv_norm_weight,
    k_pe, indexer_k,
    idx_ln_weight, idx_ln_bias,
    freqs_cis, H,
    kv_cache, pe_cache,
    idx_k_cache, idx_k_scale_cache,
    *, eps, idx_ln_dim, idx_ln_eps, rope_head_dim,
    start_pos, block_size,
    fallback,
):
    del fallback
    from partitions.dsv3_2_nvfp4.ad727876.model import (
        apply_rotary_emb, layer_norm, hadamard_transform, act_quant,
    )
    bsz, seqlen = q_c.shape[0], q_c.shape[1]
    end_pos = start_pos + seqlen

    q_c_flat = q_c.reshape(bsz * seqlen, -1).contiguous()
    kv_c_flat = kv_c.reshape(bsz * seqlen, -1).contiguous()
    qr = _rms_norm_kernel(q_c_flat, q_norm_weight.contiguous(), eps).view_as(q_c)
    kv_c_normed = _rms_norm_kernel(kv_c_flat, kv_norm_weight.contiguous(), eps).view_as(kv_c)

    k_pe_roped = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True)
    kv_cache[:bsz, start_pos:end_pos] = kv_c_normed
    pe_cache[:bsz, start_pos:end_pos] = k_pe_roped.squeeze(2)

    idx_k = layer_norm(indexer_k, idx_ln_weight, idx_ln_bias, idx_ln_dim, idx_ln_eps)
    idx_k_pe, idx_k_nope = torch.split(
        idx_k, [rope_head_dim, idx_k.shape[-1] - rope_head_dim], dim=-1,
    )
    idx_k_pe = apply_rotary_emb(idx_k_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2)
    idx_k = torch.cat([idx_k_pe, idx_k_nope], dim=-1)
    idx_k = hadamard_transform(idx_k, H)
    idx_k_fp8, idx_k_scale = act_quant(idx_k, block_size)
    idx_k_cache[:bsz, start_pos:end_pos] = idx_k_fp8.float()
    idx_k_scale_cache[:bsz, start_pos:end_pos] = idx_k_scale

    return qr
