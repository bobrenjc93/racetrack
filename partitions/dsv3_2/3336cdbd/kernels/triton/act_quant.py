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

QUANT_BLOCK = 128

if BACKEND_AVAILABLE:
    @triton.jit
    def _swiglu_quant_kernel(
        gate_ptr, up_ptr, out_fp8_ptr, out_scale_ptr,
        n_elements,
        quant_block: tl.constexpr,
    ):
        row = tl.program_id(0)
        base = row * n_elements
        for qb in range(0, tl.cdiv(n_elements, quant_block)):
            offsets = qb * quant_block + tl.arange(0, quant_block)
            mask = offsets < n_elements
            g = tl.load(gate_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(up_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
            x = (g * tl.sigmoid(g)) * u
            amax = tl.max(tl.abs(x), axis=0)
            amax = tl.where(amax > 1e-4, amax, 1e-4)
            scale = amax / 448.0
            scaled = x / scale
            clamped = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
            tl.store(out_fp8_ptr + base + offsets, clamped.to(tl.float8e4nv), mask=mask)
            tl.store(out_scale_ptr + row * tl.cdiv(n_elements, quant_block) + qb, scale)


    @triton.jit
    def _act_quant_kernel(
        x_ptr, out_fp8_ptr, out_scale_ptr,
        n_elements,
        quant_block: tl.constexpr,
    ):
        row = tl.program_id(0)
        base = row * n_elements
        for qb in range(0, tl.cdiv(n_elements, quant_block)):
            offsets = qb * quant_block + tl.arange(0, quant_block)
            mask = offsets < n_elements
            x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
            amax = tl.max(tl.abs(x), axis=0)
            amax = tl.where(amax > 1e-4, amax, 1e-4)
            scale = amax / 448.0
            scaled = x / scale
            clamped = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
            tl.store(out_fp8_ptr + base + offsets, clamped.to(tl.float8e4nv), mask=mask)
            tl.store(out_scale_ptr + row * tl.cdiv(n_elements, quant_block) + qb, scale)


def fused_act_quant(x, *, fallback):
    del fallback
    x_c = x.contiguous()
    shape = x_c.shape
    N = shape[-1]
    n_rows = x_c.numel() // N
    n_groups = N // QUANT_BLOCK
    out_fp8 = torch.empty(shape, dtype=torch.float8_e4m3fn, device=x.device)
    out_scale = torch.empty(*shape[:-1], n_groups, dtype=torch.float32, device=x.device)
    _act_quant_kernel[(n_rows,)](x_c, out_fp8, out_scale, N, QUANT_BLOCK)
    return out_fp8, out_scale


def fused_swiglu_quant(gate, up, *, fallback):
    del fallback
    gate_c = gate.contiguous()
    up_c = up.contiguous()
    shape = gate_c.shape
    N = shape[-1]
    n_rows = gate_c.numel() // N
    n_groups = N // QUANT_BLOCK
    out_fp8 = torch.empty(shape, dtype=torch.float8_e4m3fn, device=gate.device)
    out_scale = torch.empty(*shape[:-1], n_groups, dtype=torch.float32, device=gate.device)
    _swiglu_quant_kernel[(n_rows,)](gate_c, up_c, out_fp8, out_scale, N, QUANT_BLOCK)
    return out_fp8, out_scale
