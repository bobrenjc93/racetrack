from __future__ import annotations

import os

import torch

try:
    import helion
    import helion.language as hl

    BACKEND_AVAILABLE = True
except Exception:
    helion = None
    hl = None
    BACKEND_AVAILABLE = False

QUANT_BLOCK = 128


if BACKEND_AVAILABLE:

    @helion.kernel(config=helion.Config(block_sizes=[4], num_warps=2, num_stages=1), static_shapes=True)
    def _act_quant_kernel(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x shape: (n_blocks, 128) — each row is one quant block."""
        n_blocks, _block_size = x.size()
        out_fp8 = torch.empty(n_blocks, _block_size, dtype=torch.float8_e4m3fn, device=x.device)
        out_scale = torch.empty(n_blocks, dtype=torch.float32, device=x.device)
        for tile_b in hl.tile(n_blocks):
            block = x[tile_b, :].to(torch.float32)
            amax = torch.clamp(torch.amax(torch.abs(block), dim=-1), min=1e-4)
            scale = amax / 448.0
            scaled = block / scale.unsqueeze(-1)
            clamped = torch.clamp(scaled, -448.0, 448.0)
            out_fp8[tile_b, :] = clamped.to(torch.float8_e4m3fn)
            out_scale[tile_b] = scale
        return out_fp8, out_scale


def fused_act_quant(x, *, fallback):
    del fallback
    x_c = x if x.is_contiguous() else x.contiguous()
    shape = x_c.shape
    N = shape[-1]
    n_rows = x_c.numel() // N
    n_groups = (N + QUANT_BLOCK - 1) // QUANT_BLOCK
    N_padded = n_groups * QUANT_BLOCK

    x_2d = x_c.float().reshape(n_rows, N)
    if N_padded != N:
        # The helion kernel reshapes into exact (n_blocks, 128) rows, so a tail
        # smaller than QUANT_BLOCK cannot be expressed as a view. Pad the last
        # dim with zeros to mirror the triton kernel's masked load (other=0.0):
        # zeros never raise amax and the padded fp8 columns are sliced off below.
        x_2d = torch.nn.functional.pad(x_2d, (0, N_padded - N))

    x_flat = x_2d.reshape(n_rows * n_groups, QUANT_BLOCK)
    fp8_flat, scale_flat = _act_quant_kernel(x_flat)
    fp8 = fp8_flat.reshape(n_rows, N_padded)[:, :N].reshape(shape)
    return fp8, scale_flat.view(*shape[:-1], n_groups)


def fused_swiglu_quant(gate, up, *, fallback):
    del fallback
    g = gate.contiguous().float()
    u = up.contiguous().float()
    h = (g * torch.sigmoid(g)) * u
    return fused_act_quant(h.to(gate.dtype), fallback=None)
