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
            grid=[n_blocks, 1, 1], block=[BLOCK_SIZE, 1, 1], stream=stream,
        )


def fused_swiglu(gate, up, *, fallback):
    if gate.shape[0] < MIN_CUTE_ROWS:
        return fallback(gate, up)
    gate_c = gate.contiguous()
    up_c = up.contiguous()
    out = torch.empty_like(gate_c)
    rows, cols = gate_c.shape
    n_elements = rows * cols
    key = ("swiglu", str(gate_c.device), gate_c.dtype, rows, cols)
    if key not in _COMPILE_CACHE:
        stream = _stream(gate_c.device)
        _COMPILE_CACHE[key] = cute.compile(
            _swiglu_host,
            _cute_tensor(gate_c), _cute_tensor(up_c), _cute_tensor(out),
            cutlass.Int32(n_elements), cutlass.Int32(cols), stream,
        )
    stream = _stream(gate_c.device)
    _COMPILE_CACHE[key](
        _cute_tensor(gate_c), _cute_tensor(up_c), _cute_tensor(out),
        cutlass.Int32(n_elements), cutlass.Int32(cols), stream,
    )
    return out
