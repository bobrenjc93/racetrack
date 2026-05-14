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


_WEIGHT_CACHE: dict[int, Any] = {}


def _cute_tensor(tensor: torch.Tensor):
    return from_dlpack(tensor.detach()).mark_layout_dynamic()


def _cute_weight(tensor: torch.Tensor):
    key = id(tensor)
    if key not in _WEIGHT_CACHE:
        _WEIGHT_CACHE[key] = from_dlpack(tensor.detach()).mark_layout_dynamic()
    return _WEIGHT_CACHE[key]


_STREAM_CACHE: dict[int, Any] = {}


def _stream(device: torch.device):
    stream_id = torch.cuda.current_stream(device).cuda_stream
    if stream_id not in _STREAM_CACHE:
        _STREAM_CACHE[stream_id] = cuda.CUstream(stream_id)
    return _STREAM_CACHE[stream_id]


if BACKEND_AVAILABLE:

    BLOCK_SIZE = 1024
    WARP_SIZE = 32
    N_WARPS = BLOCK_SIZE // WARP_SIZE

    @cute.jit
    def _block_reduce_sum(val, tid, smem):
        val = cute.arch.warp_reduction_sum(val)
        warp_id = tid // WARP_SIZE
        lane_id = tid - warp_id * WARP_SIZE
        if lane_id == 0:
            cute.arch.store(smem + warp_id, val)
        cute.arch.sync_threads()
        if warp_id == 0:
            val = cute.arch.load(smem + lane_id, cutlass.Float32)
            val = cute.arch.warp_reduction_sum(val)
            if lane_id == 0:
                cute.arch.store(smem, val)
        cute.arch.sync_threads()
        return cute.arch.load(smem, cutlass.Float32)

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
        smem = cute.arch.alloc_smem(cutlass.Float32, N_WARPS)
        if row < rows:
            local_sum = cutlass.Float32(0.0)
            col = tid
            while col < cols:
                value = x[row, col].to(cutlass.Float32)
                local_sum += value * value
                col += BLOCK_SIZE
            total = _block_reduce_sum(local_sum, tid, smem)
            scale = cute.math.rsqrt(total / cols + eps)
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
            block=[BLOCK_SIZE, 1, 1],
            stream=stream,
        )

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
        smem = cute.arch.alloc_smem(cutlass.Float32, N_WARPS)
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
            total = _block_reduce_sum(local_sum, tid, smem)
            scale = cute.math.rsqrt(total / cols + eps)
            col = tid
            while col < cols:
                hidden = out_hidden[row, col].to(cutlass.Float32)
                w = weight[col].to(cutlass.Float32)
                out_normed[row, col] = (hidden * scale * w).to(out_normed.element_type)
                col += BLOCK_SIZE

    @cute.jit
    def _silu(value):
        return value / (cutlass.Float32(1.0) + cute.math.exp(-value))

    @cute.kernel
    def _swiglu_kernel(
        gate: cute.Tensor,
        up: cute.Tensor,
        out: cute.Tensor,
        n_elements: cutlass.Int32,
        cols: cutlass.Int32,
    ):
        bid, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        idx = bid * BLOCK_SIZE + tid
        if idx < n_elements:
            row = idx // cols
            col = idx - row * cols
            gate_value = gate[row, col].to(cutlass.Float32)
            up_value = up[row, col].to(cutlass.Float32)
            out[row, col] = (_silu(gate_value) * up_value).to(out.element_type)

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
            grid=[rows, 1, 1],
            block=[BLOCK_SIZE, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _swiglu_host(
        gate: cute.Tensor,
        up: cute.Tensor,
        out: cute.Tensor,
        n_elements: cutlass.Int32,
        cols: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        _swiglu_kernel(gate, up, out, n_elements, cols).launch(
            grid=[n_blocks, 1, 1],
            block=[BLOCK_SIZE, 1, 1],
            stream=stream,
        )


def _compiled_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
):
    _cutlass_dtype(x.dtype)
    rows, cols = x.shape
    key = ("rms_norm_v2", str(x.device), x.dtype, rows, cols)
    if key not in _COMPILE_CACHE:
        stream = _stream(x.device)
        _COMPILE_CACHE[key] = cute.compile(
            _rms_norm_host,
            _cute_tensor(x),
            _cute_weight(weight),
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


def _compiled_residual_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    out_hidden: torch.Tensor,
    out_normed: torch.Tensor,
):
    _cutlass_dtype(residual.dtype)
    rows, cols = residual.shape
    key = ("fused_residual_norm_v2", str(residual.device), residual.dtype, rows, cols)
    if key not in _COMPILE_CACHE:
        stream = _stream(residual.device)
        _COMPILE_CACHE[key] = cute.compile(
            _residual_norm_host,
            _cute_tensor(residual),
            _cute_tensor(update),
            _cute_weight(weight),
            _cute_tensor(out_hidden),
            _cute_tensor(out_normed),
            cutlass.Int32(rows),
            cutlass.Int32(cols),
            cutlass.Float32(1.0e-6),
            stream,
        )
    return _COMPILE_CACHE[key]


def _compiled_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    out: torch.Tensor,
):
    _cutlass_dtype(gate.dtype)
    rows, cols = gate.shape
    n_elements = rows * cols
    key = ("fused_swiglu_v2", str(gate.device), gate.dtype, rows, cols)
    if key not in _COMPILE_CACHE:
        stream = _stream(gate.device)
        _COMPILE_CACHE[key] = cute.compile(
            _swiglu_host,
            _cute_tensor(gate),
            _cute_tensor(up),
            _cute_tensor(out),
            cutlass.Int32(n_elements),
            cutlass.Int32(cols),
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
        _cute_weight(weight),
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


def fused_residual_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float,
    fallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    del fallback
    _require_cutedsl_cuda(residual, update, norm_weight)
    if residual.dim() != 2 or update.dim() != 2:
        raise RuntimeError("CUTEDSL fused_residual_norm expects 2D tensors")
    if residual.shape != update.shape:
        raise RuntimeError("CUTEDSL residual and update shapes must match")
    if norm_weight.dim() != 1 or norm_weight.shape[0] != residual.shape[-1]:
        raise RuntimeError("CUTEDSL norm weight shape must match hidden dimension")
    residual = residual.contiguous()
    update = update.contiguous()
    norm_weight = norm_weight.contiguous()
    out_hidden = torch.empty_like(residual)
    out_normed = torch.empty_like(residual)
    rows, cols = residual.shape
    compiled = _compiled_residual_norm(
        residual,
        update,
        norm_weight,
        out_hidden,
        out_normed,
    )
    stream = _stream(residual.device)
    compiled(
        _cute_tensor(residual),
        _cute_tensor(update),
        _cute_weight(norm_weight),
        _cute_tensor(out_hidden),
        _cute_tensor(out_normed),
        cutlass.Int32(rows),
        cutlass.Int32(cols),
        cutlass.Float32(float(eps)),
        stream,
    )
    return out_hidden, out_normed


def fused_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    fallback,
) -> torch.Tensor:
    del fallback
    _require_cutedsl_cuda(gate, up)
    if gate.dim() != 2 or up.dim() != 2:
        raise RuntimeError("CUTEDSL fused_swiglu expects 2D tensors")
    if gate.shape != up.shape:
        raise RuntimeError("CUTEDSL fused_swiglu inputs must have matching shapes")
    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate)
    rows, cols = gate.shape
    n_elements = rows * cols
    compiled = _compiled_swiglu(gate, up, out)
    stream = _stream(gate.device)
    compiled(
        _cute_tensor(gate),
        _cute_tensor(up),
        _cute_tensor(out),
        cutlass.Int32(n_elements),
        cutlass.Int32(cols),
        stream,
    )
    return out
