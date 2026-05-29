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
_FLOAT_TENSOR_CACHE = {}
BLOCK_SIZE = 256


def _cute_tensor(tensor: torch.Tensor):
    return from_dlpack(tensor.detach()).mark_layout_dynamic()


def _stream(device: torch.device):
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _cached_float_tensor(tensor):
    if tensor.dtype == torch.float32:
        return tensor
    if torch.is_grad_enabled():
        return tensor.float()
    key = (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        getattr(tensor, "_version", 0),
    )
    cached = _FLOAT_TENSOR_CACHE.get(key)
    if cached is None:
        cached = tensor.float().contiguous()
        _FLOAT_TENSOR_CACHE[key] = cached
    return cached


def _layer_norm(x, weight, bias, dim, eps):
    return F.layer_norm(
        x.float(),
        (dim,),
        _cached_float_tensor(weight),
        _cached_float_tensor(bias),
        eps,
    ).type_as(x)


def _hadamard_transform(x, H):
    d = x.shape[-1]
    return (x.float() @ _cached_float_tensor(H)[:d, :d] * (d ** -0.5)).type_as(x)


def _act_quant_unit_scale(x, block_size, act_quant):
    if x.size(-1) % block_size == 0:
        return act_quant(x, block_size)
    try:
        return x.contiguous().to(torch.float8_e4m3fn), None
    except RuntimeError:
        return x.float(), None


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

    @cute.jit
    def _rms_norm_host(
        x: cute.Tensor, weight: cute.Tensor, out: cute.Tensor,
        rows: cutlass.Int32, cols: cutlass.Int32, eps: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        _rms_norm_kernel(x, weight, out, rows, cols, eps).launch(
            grid=[rows, 1, 1], block=[BLOCK_SIZE, 1, 1], stream=stream,
        )


def _cutedsl_rms_norm(x, weight, eps):
    x_c = x.contiguous()
    weight_c = weight.contiguous()
    rows = x_c.shape[0] * (x_c.shape[1] if x_c.dim() == 3 else 1)
    cols = x_c.shape[-1]
    x_flat = x_c.reshape(rows, cols)
    out = torch.empty_like(x_flat)
    key = ("rms_norm", str(x_c.device), x_c.dtype, rows, cols)
    if key not in _COMPILE_CACHE:
        stream = _stream(x_c.device)
        _COMPILE_CACHE[key] = cute.compile(
            _rms_norm_host,
            _cute_tensor(x_flat), _cute_tensor(weight_c), _cute_tensor(out),
            cutlass.Int32(rows), cutlass.Int32(cols),
            cutlass.Float32(eps), stream,
        )
    stream = _stream(x_c.device)
    _COMPILE_CACHE[key](
        _cute_tensor(x_flat), _cute_tensor(weight_c), _cute_tensor(out),
        cutlass.Int32(rows), cutlass.Int32(cols),
        cutlass.Float32(eps), stream,
    )
    return out.view_as(x_c)


def fused_indexer_k_path(
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
    from partitions.dsv3_2_nvfp4.model import (
        apply_rotary_emb, act_quant,
    )
    bsz, seqlen = q_c.shape[0], q_c.shape[1]
    end_pos = start_pos + seqlen

    qr = _cutedsl_rms_norm(q_c, q_norm_weight, eps)
    kv_c_normed = _cutedsl_rms_norm(kv_c, kv_norm_weight, eps)

    k_pe_roped = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True)
    kv_cache[:bsz, start_pos:end_pos] = kv_c_normed
    pe_cache[:bsz, start_pos:end_pos] = k_pe_roped.squeeze(2)

    idx_k = _layer_norm(indexer_k, idx_ln_weight, idx_ln_bias, idx_ln_dim, idx_ln_eps)
    idx_k_pe, idx_k_nope = torch.split(
        idx_k, [rope_head_dim, idx_k.shape[-1] - rope_head_dim], dim=-1,
    )
    idx_k_pe = apply_rotary_emb(idx_k_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2)
    idx_k = torch.cat([idx_k_pe, idx_k_nope], dim=-1)
    idx_k = _hadamard_transform(idx_k, H)
    idx_k_fp8, idx_k_scale = _act_quant_unit_scale(idx_k, block_size, act_quant)
    idx_k_cache[:bsz, start_pos:end_pos] = idx_k_fp8.float()
    if idx_k_scale is None:
        idx_k_scale_cache[:bsz, start_pos:end_pos] = 1.0
    else:
        idx_k_scale_cache[:bsz, start_pos:end_pos] = idx_k_scale

    return qr
