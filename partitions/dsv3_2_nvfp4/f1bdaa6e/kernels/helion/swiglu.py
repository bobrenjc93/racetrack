from __future__ import annotations


import torch

try:
    import helion
    import helion.language as hl

    BACKEND_AVAILABLE = True
except Exception:
    helion = None
    hl = None
    BACKEND_AVAILABLE = False




if BACKEND_AVAILABLE:

    @helion.kernel(config=helion.Config(
            block_sizes=[1, 1024], flatten_loops=[True],
            indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor'],
            l2_groupings=[8],
            load_eviction_policies=['last', 'first'],
            loop_orders=[[0, 1]], num_stages=4, num_warps=4, pid_type='xyz',
            range_flattens=[None], range_multi_buffers=[None],
            range_num_stages=[0], range_unroll_factors=[0],
            range_warp_specializes=[],
        )
    )
    def _swiglu_kernel(
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(gate)
        rows, cols = gate.size()
        for tile_r, tile_c in hl.tile([rows, cols]):
            g = gate[tile_r, tile_c].to(torch.float32)
            u = up[tile_r, tile_c].to(torch.float32)
            silu_g = g * torch.sigmoid(g)
            out[tile_r, tile_c] = (silu_g * u).to(gate.dtype)
        return out


def fused_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    fallback,
) -> torch.Tensor:
    del fallback
    shape = gate.shape
    gate_2d = gate.contiguous().view(-1, shape[-1])
    up_2d = up.contiguous().view(-1, shape[-1])
    return _swiglu_kernel(gate_2d, up_2d).view(shape)
