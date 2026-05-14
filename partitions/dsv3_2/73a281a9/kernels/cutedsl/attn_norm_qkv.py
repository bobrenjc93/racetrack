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
        x: cute.Tensor,
        weight: cute.Tensor,
        out: cute.Tensor,
        rows: cutlass.Int32,
        cols: cutlass.Int32,
        eps: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        _rms_norm_kernel(x, weight, out, rows, cols, eps).launch(
            grid=[rows, 1, 1],
            block=[BLOCK_SIZE, 1, 1],
            stream=stream,
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


def fused_attn_norm_qkv(
    hidden_states, norm_weight, qkv_weight,
    *, eps, q_lora_rank, kv_lora_rank, qk_rope_head_dim, fallback,
):
    del fallback
    x = _cutedsl_rms_norm(hidden_states, norm_weight, eps)
    qkv = F.linear(x, qkv_weight)
    q_c, kv_c, k_pe = qkv.split(
        [q_lora_rank, kv_lora_rank, qk_rope_head_dim], dim=-1,
    )
    return q_c, kv_c, k_pe
