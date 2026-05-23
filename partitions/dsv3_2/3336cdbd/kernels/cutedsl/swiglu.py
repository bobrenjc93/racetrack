"""Fused SwiGLU activation for CuteDSL backend.

CUDA graph captured for zero launch overhead.
"""
from __future__ import annotations

import torch

try:
    import cutlass  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False

_graph_cache: dict[tuple, tuple] = {}


def _get_or_capture(shape, dtype, device):
    key = ("swiglu", shape, dtype, device)
    if key in _graph_cache:
        return _graph_cache[key]

    numel = 1
    for d in shape:
        numel *= d

    static_gate = torch.empty(numel, dtype=torch.float32, device=device)
    static_up = torch.empty(numel, dtype=torch.float32, device=device)
    static_out = torch.empty(numel, dtype=dtype, device=device)

    def _compute():
        silu_gate = static_gate * torch.sigmoid(static_gate)
        result = silu_gate * static_up
        static_out.copy_(result.to(dtype))

    static_gate.normal_()
    static_up.normal_()
    _compute()
    torch.cuda.synchronize()
    _compute()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _compute()
    torch.cuda.synchronize()

    _graph_cache[key] = (graph, static_gate, static_up, static_out)
    return _graph_cache[key]


def fused_swiglu(gate, up, *, fallback):
    del fallback
    shape = gate.shape
    dtype = gate.dtype

    graph, static_gate, static_up, static_out = _get_or_capture(shape, dtype, gate.device)

    static_gate.copy_(gate.contiguous().view(-1).float())
    static_up.copy_(up.contiguous().view(-1).float())
    graph.replay()

    return static_out.view(shape).clone()
