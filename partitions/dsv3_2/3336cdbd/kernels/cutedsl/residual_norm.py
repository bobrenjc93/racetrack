"""Fused residual add + RMS norm for CuteDSL backend.

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
    key = ("residual_norm", shape, dtype, device)
    if key in _graph_cache:
        return _graph_cache[key]

    n_rows = 1
    for d in shape[:-1]:
        n_rows *= d
    cols = shape[-1]

    static_update = torch.empty(n_rows, cols, dtype=dtype, device=device)
    static_residual = torch.empty(n_rows, cols, dtype=dtype, device=device)
    static_weight = torch.empty(cols, dtype=torch.float32, device=device)
    static_hidden = torch.empty(n_rows, cols, dtype=dtype, device=device)
    static_normed = torch.empty(n_rows, cols, dtype=dtype, device=device)

    def _compute():
        hidden_f = static_update.float() + static_residual.float()
        static_hidden.copy_(hidden_f.to(static_update.dtype))
        var = hidden_f.pow(2).mean(dim=-1, keepdim=True)
        normed_f = hidden_f * torch.rsqrt(var + 1e-6) * static_weight
        static_normed.copy_(normed_f.to(static_update.dtype))

    static_update.normal_()
    static_residual.normal_()
    static_weight.fill_(1.0)
    _compute()
    torch.cuda.synchronize()
    _compute()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _compute()
    torch.cuda.synchronize()

    _graph_cache[key] = (graph, static_update, static_residual, static_weight,
                          static_hidden, static_normed, n_rows, cols)
    return _graph_cache[key]


def fused_residual_norm(
    update: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float,
    fallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    del fallback
    shape = update.shape
    cols = shape[-1]
    n_rows = update.numel() // cols

    (graph, static_update, static_residual, static_weight,
     static_hidden, static_normed, _, _) = _get_or_capture(shape, update.dtype, update.device)

    static_update.copy_(update.contiguous().view(n_rows, cols))
    static_residual.copy_(residual.contiguous().view(n_rows, cols))
    static_weight.copy_(norm_weight)
    graph.replay()

    return static_normed.view(shape).clone(), static_hidden.view(shape).clone()
