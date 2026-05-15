from __future__ import annotations

import torch
import torch.distributed as dist

try:
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False


def fused_single_token_moe(
    moe,
    x: torch.Tensor,
    *,
    fallback,
) -> torch.Tensor:
    shape = x.size()
    flat = x.view(-1, moe.dim)
    if flat.shape[0] != 1:
        return fallback(moe, x)

    from inference import model as real_model

    weights, indices = moe.gate(flat)
    y = torch.zeros_like(flat, dtype=torch.float32)
    for top, expert_id in enumerate(indices[0].tolist()):
        if moe.experts_start_idx <= expert_id < moe.experts_end_idx:
            y += moe.experts[expert_id](flat) * weights[0, top, None]
    y += moe.shared_experts(flat)
    if real_model.world_size > 1:
        dist.all_reduce(y)
    return y.type_as(flat).view(shape)
