from __future__ import annotations

from typing import Any

import torch

from . import torch_ops

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


def _require_cutedsl_cuda(*tensors: torch.Tensor) -> None:
    if not BACKEND_AVAILABLE:
        raise RuntimeError("CUTEDSL backend requested, but nvidia-cutlass-dsl is not installed")
    if not tensors or any(tensor.device.type != "cuda" for tensor in tensors):
        raise RuntimeError("CUTEDSL kernels require CUDA tensors")


def _cutlass_dtype(dtype: torch.dtype):
    if dtype is torch.float16:
        return cutlass.Float16
    if dtype is torch.bfloat16:
        return cutlass.BFloat16
    if dtype is torch.float32:
        return cutlass.Float32
    raise RuntimeError(f"CUTEDSL backend does not support dtype {dtype}")


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
        if row < rows:
            sum_sq = cutlass.Float32(0.0)
            for col in cutlass.range(cols):
                value = x[row, col].to(cutlass.Float32)
                sum_sq += value * value
            scale = cute.math.rsqrt(sum_sq / cols + eps)
            for col in cutlass.range(cols):
                value = x[row, col].to(cutlass.Float32)
                w = weight[col].to(cutlass.Float32)
                out[row, col] = (value * scale * w).to(out.element_type)

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
            block=[1, 1, 1],
            stream=stream,
        )

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
        if row < rows:
            half = rotary_dim // 2
            for col in cutlass.range(half):
                c = cos[row, col]
                s = sin[row, col]
                x1 = x[row, col]
                x2 = x[row, col + half]
                out[row, col] = x1 * c - x2 * s
                out[row, col + half] = x2 * c + x1 * s

    @cute.jit
    def _rope_host(
        x: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
        rows: cutlass.Int32,
        rotary_dim: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _rope_kernel(x, cos, sin, out, rows, rotary_dim).launch(
            grid=[rows, 1, 1],
            block=[1, 1, 1],
            stream=stream,
        )


def _compiled_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
):
    _cutlass_dtype(x.dtype)
    rows, cols = x.shape
    key = ("rms_norm", str(x.device), x.dtype, rows, cols)
    if key not in _COMPILE_CACHE:
        stream = _stream(x.device)
        _COMPILE_CACHE[key] = cute.compile(
            _rms_norm_host,
            _cute_tensor(x),
            _cute_tensor(weight),
            _cute_tensor(out),
            cutlass.Int32(rows),
            cutlass.Int32(cols),
            cutlass.Float32(1.0e-6),
            stream,
        )
    return _COMPILE_CACHE[key]


def _compiled_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: torch.Tensor,
):
    _cutlass_dtype(x.dtype)
    rows, rotary_dim = x.shape
    key = ("rope", str(x.device), x.dtype, rows, rotary_dim)
    if key not in _COMPILE_CACHE:
        stream = _stream(x.device)
        _COMPILE_CACHE[key] = cute.compile(
            _rope_host,
            _cute_tensor(x),
            _cute_tensor(cos),
            _cute_tensor(sin),
            _cute_tensor(out),
            cutlass.Int32(rows),
            cutlass.Int32(rotary_dim),
            stream,
        )
    return _COMPILE_CACHE[key]


def _rms_norm_cutedsl(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    _require_cutedsl_cuda(x, weight)
    if x.dim() != 2:
        raise RuntimeError("CUTEDSL RMSNorm kernel expects a 2D tensor")
    if weight.dim() != 1 or weight.shape[0] != x.shape[-1]:
        raise RuntimeError("CUTEDSL RMSNorm weight shape must match the hidden dimension")
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty_like(x)
    rows, cols = x.shape
    compiled = _compiled_rms_norm(x, weight, out)
    stream = _stream(x.device)
    compiled(
        _cute_tensor(x),
        _cute_tensor(weight),
        _cute_tensor(out),
        cutlass.Int32(rows),
        cutlass.Int32(cols),
        cutlass.Float32(float(eps)),
        stream,
    )
    return out


def _apply_rope_cutedsl(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_base: float,
) -> torch.Tensor:
    _require_cutedsl_cuda(x, positions)
    if x.dim() != 2:
        raise RuntimeError("CUTEDSL RoPE kernel expects a 2D tensor")
    if x.shape[-1] % 2 != 0:
        raise RuntimeError("CUTEDSL RoPE kernel requires an even RoPE dimension")
    x = x.contiguous()
    positions = positions.contiguous()
    cos, sin = torch_ops.rope_cache(
        positions,
        x.shape[-1],
        base=rope_base,
        dtype=x.dtype,
    )
    cos = cos.contiguous()
    sin = sin.contiguous()
    out = torch.empty_like(x)
    rows, rotary_dim = x.shape
    compiled = _compiled_rope(x, cos, sin, out)
    stream = _stream(x.device)
    compiled(
        _cute_tensor(x),
        _cute_tensor(cos),
        _cute_tensor(sin),
        _cute_tensor(out),
        cutlass.Int32(rows),
        cutlass.Int32(rotary_dim),
        stream,
    )
    return out


def fused_norm_rope(
    q_c: torch.Tensor,
    q_weight: torch.Tensor,
    kv_c: torch.Tensor,
    kv_weight: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
    *,
    eps: float,
    rope_base: float,
    fallback,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del fallback
    _require_cutedsl_cuda(q_c, q_weight, kv_c, kv_weight, k_pe, positions)
    return (
        _rms_norm_cutedsl(q_c, q_weight, eps),
        _rms_norm_cutedsl(kv_c, kv_weight, eps),
        _apply_rope_cutedsl(k_pe, positions, rope_base=rope_base),
    )
