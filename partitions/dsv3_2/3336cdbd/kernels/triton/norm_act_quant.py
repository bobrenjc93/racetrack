"""Fused RMSNorm + FP8 act_quant in a single Triton kernel.

Inductor generates this as one kernel: read bf16 input once,
compute norm + weight multiplication + per-block amax + scale +
fp8 quantization, write fp8 + scale output. Never materializes
the bf16 normed intermediate.

For residual norm: also writes the hidden (residual + update) state.
"""
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
    def _residual_norm_quant_kernel(
        residual_ptr, update_ptr, weight_ptr,
        out_hidden_ptr, out_fp8_ptr, out_scale_ptr,
        eps: tl.constexpr, cols: tl.constexpr,
        quant_block: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """
        One kernel: residual_add + rmsnorm + per-block fp8 quantization.
        Reads input once, writes hidden (bf16) + fp8 + scale.
        """
        token = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < cols

        r = tl.load(residual_ptr + token * cols + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(update_ptr + token * cols + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Residual add
        hidden = r + u
        tl.store(out_hidden_ptr + token * cols + offsets, hidden, mask=mask)

        # RMSNorm
        variance = tl.sum(hidden * hidden, axis=0) / cols
        inv_rms = tl.rsqrt(variance + eps)

        # Norm + quantize per block (no intermediate bf16 write)
        n_groups = cols // quant_block
        for qb in range(0, n_groups):
            qb_start = qb * quant_block
            qb_offsets = qb_start + tl.arange(0, quant_block)
            qb_mask = qb_offsets < cols

            h = tl.load(out_hidden_ptr + token * cols + qb_offsets, mask=qb_mask, other=0.0).to(tl.float32)
            w_block = tl.load(weight_ptr + qb_offsets, mask=qb_mask, other=0.0).to(tl.float32)
            normed = h * inv_rms * w_block

            # Per-block FP8 quantization
            amax = tl.max(tl.abs(normed), axis=0)
            amax = tl.where(amax > 1e-4, amax, 1e-4)
            scale = amax / 448.0
            scaled = normed / scale
            clamped = tl.minimum(tl.maximum(scaled, -448.0), 448.0)

            tl.store(out_fp8_ptr + token * cols + qb_offsets, clamped.to(tl.float8e4nv), mask=qb_mask)
            tl.store(out_scale_ptr + token * n_groups + qb, scale)

    @triton.jit
    def _standalone_norm_quant_kernel(
        x_ptr, weight_ptr,
        out_fp8_ptr, out_scale_ptr,
        eps: tl.constexpr, cols: tl.constexpr,
        quant_block: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """
        Standalone RMSNorm + FP8 quant (no residual add).
        """
        token = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < cols

        x = tl.load(x_ptr + token * cols + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / cols
        inv_rms = tl.rsqrt(variance + eps)

        n_groups = cols // quant_block
        for qb in range(0, n_groups):
            qb_start = qb * quant_block
            qb_offsets = qb_start + tl.arange(0, quant_block)
            qb_mask = qb_offsets < cols

            xb = tl.load(x_ptr + token * cols + qb_offsets, mask=qb_mask, other=0.0).to(tl.float32)
            wb = tl.load(weight_ptr + qb_offsets, mask=qb_mask, other=0.0).to(tl.float32)
            normed = xb * inv_rms * wb

            amax = tl.max(tl.abs(normed), axis=0)
            amax = tl.where(amax > 1e-4, amax, 1e-4)
            scale = amax / 448.0
            scaled = normed / scale
            clamped = tl.minimum(tl.maximum(scaled, -448.0), 448.0)

            tl.store(out_fp8_ptr + token * cols + qb_offsets, clamped.to(tl.float8e4nv), mask=qb_mask)
            tl.store(out_scale_ptr + token * n_groups + qb, scale)


def fused_residual_norm_quant(
    residual, update, weight, *, eps,
):
    """Returns (hidden_bf16, normed_fp8, normed_scale)."""
    shape = update.shape
    cols = shape[-1]
    n_tokens = update.numel() // cols
    n_groups = cols // QUANT_BLOCK

    out_hidden = torch.empty_like(update)
    out_fp8 = torch.empty(shape, dtype=torch.float8_e4m3fn, device=update.device)
    out_scale = torch.empty(*shape[:-1], n_groups, dtype=torch.float32, device=update.device)

    block_size = triton.next_power_of_2(cols)
    _residual_norm_quant_kernel[(n_tokens,)](
        residual.contiguous(), update.contiguous(), weight,
        out_hidden, out_fp8, out_scale,
        eps, cols, QUANT_BLOCK, block_size,
    )
    return out_hidden, out_fp8, out_scale


def fused_standalone_norm_quant(x, weight, *, eps):
    """Returns (normed_fp8, normed_scale)."""
    shape = x.shape
    cols = shape[-1]
    n_tokens = x.numel() // cols
    n_groups = cols // QUANT_BLOCK

    out_fp8 = torch.empty(shape, dtype=torch.float8_e4m3fn, device=x.device)
    out_scale = torch.empty(*shape[:-1], n_groups, dtype=torch.float32, device=x.device)

    block_size = triton.next_power_of_2(cols)
    _standalone_norm_quant_kernel[(n_tokens,)](
        x.contiguous(), weight,
        out_fp8, out_scale,
        eps, cols, QUANT_BLOCK, block_size,
    )
    return out_fp8, out_scale
