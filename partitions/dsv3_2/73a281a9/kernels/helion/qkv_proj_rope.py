from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import helion
    import helion.language as hl

    BACKEND_AVAILABLE = True
except Exception:
    helion = None
    hl = None
    BACKEND_AVAILABLE = False




if BACKEND_AVAILABLE:

    @helion.kernel(config=helion.Config(
            block_sizes=[1],
            indexing=['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor',
                      'tensor_descriptor', 'pointer', 'pointer', 'pointer'],
            load_eviction_policies=['last', '', 'first', 'last', 'last'],
            num_stages=8, num_warps=32, pid_type='flat',
            range_flattens=[None], range_multi_buffers=[None],
            range_num_stages=[0], range_unroll_factors=[0],
            range_warp_specializes=[], reduction_loops=[None],
        )
    )
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

    @helion.kernel(config=helion.Config(
            block_sizes=[1, 2],
            indexing=['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor',
                      'tensor_descriptor', 'tensor_descriptor'],
            l2_groupings=[16],
            load_eviction_policies=['first', 'first', 'first'],
            loop_orders=[[0, 1]], num_stages=7, num_warps=4, pid_type='flat',
            range_flattens=[None], range_multi_buffers=[None],
            range_num_stages=[0], range_unroll_factors=[0],
            range_warp_specializes=[],
        )
    )
    def _rope_kernel(
        x: torch.Tensor,
        positions: torch.Tensor,
        log_rope_base: float,
    ) -> torch.Tensor:
        out = torch.empty_like(x)
        tokens, rotary_dim = x.size()
        half = rotary_dim // 2
        for tile_t, tile_h in hl.tile([tokens, half]):
            rotary_index = tile_h.index.to(torch.float32)
            position_values = positions[tile_t].to(torch.float32).view(tile_t, 1)
            inv_freq = torch.exp(-(rotary_index / half) * log_rope_base)
            freqs = position_values * inv_freq.view(1, tile_h)
            cos = torch.cos(freqs).to(x.dtype)
            sin = torch.sin(freqs).to(x.dtype)
            x1 = x[tile_t, tile_h]
            x2 = x[tile_t, tile_h + half]
            out[tile_t, tile_h] = (x1 * cos - x2 * sin).to(x.dtype)
            out[tile_t, tile_h + half] = (x2 * cos + x1 * sin).to(x.dtype)
        return out


def fused_qkv_proj_rope(
    q_c, q_norm_weight, q_b_weight,
    kv_c, kv_norm_weight, kv_b_weight,
    k_pe, positions,
    *, eps, num_heads, head_dim, nope_dim, rope_dim, v_head_dim, rope_base,
    fallback,
):
    del fallback
    tokens = q_c.shape[0]
    log_base = math.log(rope_base)

    q_c_2d = q_c.contiguous().view(-1, q_c.shape[-1])
    q_c = _rms_norm_kernel(q_c_2d, q_norm_weight.contiguous(), eps).view_as(q_c)
    q = F.linear(q_c, q_b_weight).view(tokens, num_heads, head_dim)
    q_nope, q_pe = q.split([nope_dim, rope_dim], dim=-1)
    q_pe_flat = q_pe.reshape(-1, rope_dim).contiguous()
    pos_expanded = positions.unsqueeze(1).expand(-1, num_heads).reshape(-1).contiguous()
    q_pe_roped = _rope_kernel(q_pe_flat, pos_expanded, log_base)
    q = torch.cat((q_nope, q_pe_roped.view_as(q_pe)), dim=-1)

    kv_c_2d = kv_c.contiguous().view(-1, kv_c.shape[-1])
    kv_c = _rms_norm_kernel(kv_c_2d, kv_norm_weight.contiguous(), eps).view_as(kv_c)
    k_pe_2d = k_pe.contiguous().view(-1, k_pe.shape[-1])
    k_pe = _rope_kernel(k_pe_2d, positions.contiguous(), log_base).view_as(k_pe)
    kv = F.linear(kv_c, kv_b_weight).view(tokens, num_heads, nope_dim + v_head_dim)
    k_nope, v = kv.split([nope_dim, v_head_dim], dim=-1)
    k = torch.cat((k_nope, k_pe.unsqueeze(1).expand(-1, num_heads, -1)), dim=-1)

    return q, k, v
