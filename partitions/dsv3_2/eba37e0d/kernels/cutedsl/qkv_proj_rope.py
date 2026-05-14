from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

try:
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack

    BACKEND_AVAILABLE = True
except Exception:
    cutlass = None
    cute = None
    cuda = None
    from_dlpack = None
    BACKEND_AVAILABLE = False


_COMPILE_CACHE: dict[tuple[Any, ...], Any] = {}

BLOCK_SIZE = 256


def _cute_tensor(tensor: torch.Tensor):
    return from_dlpack(tensor.detach()).mark_layout_dynamic()


def _stream(device: torch.device):
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _rope_cache(positions, rotary_dim, *, base, dtype):
    device = positions.device
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) / max(half, 1))
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


if BACKEND_AVAILABLE:

    @cute.kernel
    def _rms_norm_kernel(
        x: cute.Tensor,
        weight: cute.Tensor,
        out: cute.Tensor,
        rows: cutlass.Int32,
        cols: cutlass.Int32,
        eps: cutlass.Float32,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        smem_ptr = cute.arch.alloc_smem(cutlass.Float32, BLOCK_SIZE)
        if row < rows:
            local_sum = cutlass.Float32(0.0)
            col = tid
            while col < cols:
                value = x[row, col].to(cutlass.Float32)
                local_sum += value * value
                col += BLOCK_SIZE
            cute.arch.store(smem_ptr + tid, local_sum)
            cute.arch.sync_threads()
            stride = BLOCK_SIZE // 2
            while stride > 0:
                if tid < stride:
                    a = cute.arch.load(smem_ptr + tid, cutlass.Float32)
                    b = cute.arch.load(smem_ptr + tid + stride, cutlass.Float32)
                    cute.arch.store(smem_ptr + tid, a + b)
                cute.arch.sync_threads()
                stride = stride // 2
            total = cute.arch.load(smem_ptr, cutlass.Float32)
            scale = cute.math.rsqrt(total / cols + eps)
            cute.arch.sync_threads()
            col = tid
            while col < cols:
                value = x[row, col].to(cutlass.Float32)
                w = weight[col].to(cutlass.Float32)
                out[row, col] = (value * scale * w).to(out.element_type)
                col += BLOCK_SIZE

    @cute.kernel
    def _rope_kernel(
        x: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
        rows: cutlass.Int32,
        rotary_dim: cutlass.Int32,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        if row < rows:
            half = rotary_dim // 2
            col = tid
            while col < half:
                c = cos[row, col]
                s = sin[row, col]
                x1 = x[row, col]
                x2 = x[row, col + half]
                out[row, col] = x1 * c - x2 * s
                out[row, col + half] = x2 * c + x1 * s
                col += BLOCK_SIZE

    @cute.jit
    def _rms_norm_host(
        x: cute.Tensor, weight: cute.Tensor, out: cute.Tensor,
        rows: cutlass.Int32, cols: cutlass.Int32, eps: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        _rms_norm_kernel(x, weight, out, rows, cols, eps).launch(
            grid=[rows, 1, 1], block=[BLOCK_SIZE, 1, 1], stream=stream,
        )

    @cute.jit
    def _rope_host(
        x: cute.Tensor, cos: cute.Tensor, sin: cute.Tensor, out: cute.Tensor,
        rows: cutlass.Int32, rotary_dim: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _rope_kernel(x, cos, sin, out, rows, rotary_dim).launch(
            grid=[rows, 1, 1], block=[BLOCK_SIZE, 1, 1], stream=stream,
        )


def _cutedsl_rms_norm(x, weight, eps):
    x_c = x.contiguous()
    weight_c = weight.contiguous()
    out = torch.empty_like(x_c)
    rows, cols = x_c.shape
    key = ("rms_norm", str(x_c.device), x_c.dtype, rows, cols)
    if key not in _COMPILE_CACHE:
        stream = _stream(x_c.device)
        _COMPILE_CACHE[key] = cute.compile(
            _rms_norm_host,
            _cute_tensor(x_c), _cute_tensor(weight_c), _cute_tensor(out),
            cutlass.Int32(rows), cutlass.Int32(cols),
            cutlass.Float32(eps), stream,
        )
    stream = _stream(x_c.device)
    _COMPILE_CACHE[key](
        _cute_tensor(x_c), _cute_tensor(weight_c), _cute_tensor(out),
        cutlass.Int32(rows), cutlass.Int32(cols),
        cutlass.Float32(eps), stream,
    )
    return out


def _cutedsl_rope(x, positions, rope_base):
    x_c = x.contiguous()
    rows, rotary_dim = x_c.shape
    cos, sin = _rope_cache(positions, rotary_dim, base=rope_base, dtype=x_c.dtype)
    cos_c, sin_c = cos.contiguous(), sin.contiguous()
    out = torch.empty_like(x_c)
    key = ("rope", str(x_c.device), x_c.dtype, rows, rotary_dim)
    if key not in _COMPILE_CACHE:
        stream = _stream(x_c.device)
        _COMPILE_CACHE[key] = cute.compile(
            _rope_host,
            _cute_tensor(x_c), _cute_tensor(cos_c), _cute_tensor(sin_c),
            _cute_tensor(out),
            cutlass.Int32(rows), cutlass.Int32(rotary_dim), stream,
        )
    stream = _stream(x_c.device)
    _COMPILE_CACHE[key](
        _cute_tensor(x_c), _cute_tensor(cos_c), _cute_tensor(sin_c),
        _cute_tensor(out),
        cutlass.Int32(rows), cutlass.Int32(rotary_dim), stream,
    )
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

    q_c = _cutedsl_rms_norm(q_c, q_norm_weight, eps)
    q = F.linear(q_c, q_b_weight).view(tokens, num_heads, head_dim)
    q_nope, q_pe = q.split([nope_dim, rope_dim], dim=-1)
    q_pe_flat = q_pe.reshape(-1, rope_dim).contiguous()
    pos_expanded = positions.unsqueeze(1).expand(-1, num_heads).reshape(-1).contiguous()
    q_pe_roped = _cutedsl_rope(q_pe_flat, pos_expanded, rope_base)
    q = torch.cat((q_nope, q_pe_roped.view_as(q_pe)), dim=-1)

    kv_c = _cutedsl_rms_norm(kv_c, kv_norm_weight, eps)
    k_pe = _cutedsl_rope(k_pe, positions, rope_base)
    kv = F.linear(kv_c, kv_b_weight).view(tokens, num_heads, nope_dim + v_head_dim)
    k_nope, v = kv.split([nope_dim, v_head_dim], dim=-1)
    k = torch.cat((k_nope, k_pe.unsqueeze(1).expand(-1, num_heads, -1)), dim=-1)

    return q, k, v
