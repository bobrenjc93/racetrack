from __future__ import annotations

import torch

try:
    import helion  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False


def fused_score_softmax(
    scores_nope: torch.Tensor,
    scores_rope: torch.Tensor,
    index_mask: torch.Tensor,
    *,
    softmax_scale: float,
    fallback,
) -> torch.Tensor:
    del fallback
    orig_shape = scores_nope.shape

    combined = (scores_nope.float() + scores_rope.float()) * softmax_scale

    if index_mask.dim() == 4:
        mask = index_mask
    elif index_mask.dim() == 3:
        mask = index_mask.unsqueeze(2) if combined.dim() == 4 else index_mask
    elif index_mask.dim() == 2:
        mask = index_mask.unsqueeze(0).unsqueeze(2) if combined.dim() == 4 else index_mask.unsqueeze(0)
    else:
        mask = index_mask

    combined = combined + mask.float()
    probs = torch.softmax(combined, dim=-1).to(scores_nope.dtype)
    return probs.view(orig_shape)
