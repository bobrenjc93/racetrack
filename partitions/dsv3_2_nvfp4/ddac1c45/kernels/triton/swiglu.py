from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    BACKEND_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    BACKEND_AVAILABLE = False


if BACKEND_AVAILABLE:

    @triton.jit
    def _swiglu_kernel(
        gate, up, out,
        n_elements: tl.constexpr, block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        gate_values = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
        up_values = tl.load(up + offsets, mask=mask, other=0.0).to(tl.float32)
        silu_gate = gate_values * tl.sigmoid(gate_values)
        tl.store(out + offsets, silu_gate * up_values, mask=mask)


def fused_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    fallback,
) -> torch.Tensor:
    del fallback
    gate_c = gate.contiguous()
    up_c = up.contiguous()
    out = torch.empty_like(gate_c)
    n_elements = gate_c.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    _swiglu_kernel[grid](gate_c, up_c, out, n_elements, block_size, num_warps=4)
    return out.view_as(gate)
