from __future__ import annotations

from typing import Any

import torch

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
MIN_CUTE_ROWS = 512


def _cute_tensor(tensor: torch.Tensor):
    return from_dlpack(tensor.detach()).mark_layout_dynamic()


def _stream(device: torch.device):
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


if BACKEND_AVAILABLE:

    @cute.kernel
    def _residual_norm_kernel(
        residual: cute.Tensor,
        update: cute.Tensor,
        weight: cute.Tensor,
        out_hidden: cute.Tensor,
        out_normed: cute.Tensor,
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
                hidden = (
                    residual[row, col].to(cutlass.Float32)
                    + update[row, col].to(cutlass.Float32)
                )
                out_hidden[row, col] = hidden.to(out_hidden.element_type)
                local_sum += hidden * hidden
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
                hidden = out_hidden[row, col].to(cutlass.Float32)
                w = weight[col].to(cutlass.Float32)
                out_normed[row, col] = (hidden * scale * w).to(out_normed.element_type)
                col += BLOCK_SIZE

    @cute.jit
    def _residual_norm_host(
        residual: cute.Tensor,
        update: cute.Tensor,
        weight: cute.Tensor,
        out_hidden: cute.Tensor,
        out_normed: cute.Tensor,
        rows: cutlass.Int32,
        cols: cutlass.Int32,
        eps: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        _residual_norm_kernel(
            residual, update, weight, out_hidden, out_normed,
            rows, cols, eps,
        ).launch(
            grid=[rows, 1, 1], block=[BLOCK_SIZE, 1, 1], stream=stream,
        )


def fused_residual_norm(
    residual, update, norm_weight, *, eps, fallback,
):
    if residual.shape[0] < MIN_CUTE_ROWS:
        return fallback(residual, update, norm_weight, eps=eps)
    residual_c = residual.contiguous()
    update_c = update.contiguous()
    weight_c = norm_weight.contiguous()
    out_hidden = torch.empty_like(residual_c)
    out_normed = torch.empty_like(residual_c)
    rows, cols = residual_c.shape
    key = ("residual_norm", str(residual_c.device), residual_c.dtype, rows, cols)
    if key not in _COMPILE_CACHE:
        stream = _stream(residual_c.device)
        _COMPILE_CACHE[key] = cute.compile(
            _residual_norm_host,
            _cute_tensor(residual_c), _cute_tensor(update_c), _cute_tensor(weight_c),
            _cute_tensor(out_hidden), _cute_tensor(out_normed),
            cutlass.Int32(rows), cutlass.Int32(cols),
            cutlass.Float32(eps), stream,
        )
    stream = _stream(residual_c.device)
    _COMPILE_CACHE[key](
        _cute_tensor(residual_c), _cute_tensor(update_c), _cute_tensor(weight_c),
        _cute_tensor(out_hidden), _cute_tensor(out_normed),
        cutlass.Int32(rows), cutlass.Int32(cols),
        cutlass.Float32(eps), stream,
    )
    return out_hidden, out_normed
