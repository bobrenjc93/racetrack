"""FP8 per-block activation quantization for the CuteDSL backend.

Uses CUDA graph capture to fuse the multi-op PyTorch chain (float →
abs → amax → scale → clamp → fp8) into a single GPU-side replay,
eliminating per-op launch overhead.
"""
from __future__ import annotations

import torch

try:
    import cutlass  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False

QUANT_BLOCK = 128
_graph_cache: dict[tuple, tuple] = {}


def _get_or_capture(shape, dtype, device):
    key = (shape, dtype, device)
    if key in _graph_cache:
        return _graph_cache[key]

    N = shape[-1]
    n_rows = 1
    for d in shape[:-1]:
        n_rows *= d
    n_groups = (N + QUANT_BLOCK - 1) // QUANT_BLOCK

    static_x = torch.empty(n_rows, n_groups, QUANT_BLOCK, dtype=torch.float32, device=device)
    static_fp8 = torch.empty(n_rows, N, dtype=torch.float8_e4m3fn, device=device)
    static_scale = torch.empty(n_rows, n_groups, dtype=torch.float32, device=device)

    def _compute():
        amax = static_x.abs().amax(dim=-1).clamp(min=1e-4)
        scale = amax / 448.0
        scaled = static_x / scale.unsqueeze(-1)
        clamped = scaled.clamp(-448.0, 448.0)
        static_fp8.copy_(clamped.view(n_rows, N).to(torch.float8_e4m3fn))
        static_scale.copy_(scale)

    # Warmup
    static_x.normal_()
    _compute()
    torch.cuda.synchronize()
    _compute()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _compute()
    torch.cuda.synchronize()

    _graph_cache[key] = (graph, static_x, static_fp8, static_scale, n_rows, n_groups)
    return _graph_cache[key]


def fused_act_quant(x, *, fallback):
    del fallback
    x_c = x if x.is_contiguous() else x.contiguous()
    shape = x_c.shape
    N = shape[-1]
    n_rows = x_c.numel() // N
    n_groups = (N + QUANT_BLOCK - 1) // QUANT_BLOCK

    graph, static_x, static_fp8, static_scale, _, _ = _get_or_capture(shape, x_c.dtype, x_c.device)

    static_x.copy_(x_c.float().view(n_rows, n_groups, QUANT_BLOCK))
    graph.replay()

    return static_fp8.view(shape), static_scale.view(*shape[:-1], n_groups)


_sg_graph_cache: dict[tuple, tuple] = {}


def _get_or_capture_sg(gate_shape, dtype, device):
    key = ("swiglu_quant", gate_shape, dtype, device)
    if key in _sg_graph_cache:
        return _sg_graph_cache[key]

    N = gate_shape[-1]
    n_rows = 1
    for d in gate_shape[:-1]:
        n_rows *= d
    n_groups = (N + QUANT_BLOCK - 1) // QUANT_BLOCK

    static_gate = torch.empty(n_rows * N, dtype=torch.float32, device=device)
    static_up = torch.empty(n_rows * N, dtype=torch.float32, device=device)
    static_fp8 = torch.empty(n_rows, N, dtype=torch.float8_e4m3fn, device=device)
    static_scale = torch.empty(n_rows, n_groups, dtype=torch.float32, device=device)

    def _compute():
        silu_gate = static_gate * torch.sigmoid(static_gate)
        h = (silu_gate * static_up).view(n_rows, n_groups, QUANT_BLOCK)
        amax = h.abs().amax(dim=-1).clamp(min=1e-4)
        scale = amax / 448.0
        scaled = h / scale.unsqueeze(-1)
        clamped = scaled.clamp(-448.0, 448.0)
        static_fp8.copy_(clamped.view(n_rows, N).to(torch.float8_e4m3fn))
        static_scale.copy_(scale)

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

    _sg_graph_cache[key] = (graph, static_gate, static_up, static_fp8, static_scale)
    return _sg_graph_cache[key]


def fused_swiglu_quant(gate, up, *, fallback):
    del fallback
    shape = gate.shape

    graph, static_gate, static_up, static_fp8, static_scale = \
        _get_or_capture_sg(shape, gate.dtype, gate.device)

    static_gate.copy_(gate.contiguous().view(-1).float())
    static_up.copy_(up.contiguous().view(-1).float())
    graph.replay()

    N = shape[-1]
    n_groups = (N + QUANT_BLOCK - 1) // QUANT_BLOCK
    return static_fp8.view(shape), static_scale.view(*shape[:-1], n_groups)
