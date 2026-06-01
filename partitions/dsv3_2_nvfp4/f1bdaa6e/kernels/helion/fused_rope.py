from __future__ import annotations


import torch

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
            vals = x[tile_t, :].to(torch.float32)
            variance = torch.mean(vals * vals, dim=1)
            scale = torch.rsqrt(variance + eps).view(tile_t, 1)
            normed = vals * scale * weight[:].to(torch.float32)
            out[tile_t, :] = normed.to(x.dtype)
        return out


def _rope_cache(positions, rotary_dim, *, base, dtype):
    device = positions.device
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) / max(half, 1))
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def _apply_rope(k_pe, positions, *, rope_base):
    d = k_pe.shape[-1]
    half_d = d // 2
    cos, sin = _rope_cache(positions, d, base=rope_base, dtype=torch.float32)
    x1 = k_pe[:, :half_d].float()
    x2 = k_pe[:, half_d:].float()
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(k_pe.dtype)


def fused_norm_rope(
    q_c, q_weight, kv_c, kv_weight, k_pe, positions,
    *, eps, rope_base, fallback,
):
    del fallback
    q_shape = q_c.shape
    kv_shape = kv_c.shape
    q_2d = q_c.contiguous().view(-1, q_shape[-1])
    kv_2d = kv_c.contiguous().view(-1, kv_shape[-1])
    q_out = _rms_norm_kernel(q_2d, q_weight.contiguous(), eps).view(q_shape)
    kv_out = _rms_norm_kernel(kv_2d, kv_weight.contiguous(), eps).view(kv_shape)
    k_pe_out = _apply_rope(k_pe.contiguous(), positions, rope_base=rope_base)
    return q_out, kv_out, k_pe_out
