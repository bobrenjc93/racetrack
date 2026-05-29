from __future__ import annotations

import torch

try:
    import helion  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False


def fused_full_topk_indexer(
    indexer,
    x: torch.Tensor,
    qr: torch.Tensor,
    start_pos: int,
    freqs_cis: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    fallback,
) -> torch.Tensor:
    # The full-topk patcher (_patch_full_topk_indexer) short-circuits and writes
    # the indexer K-cache itself for every position with end_pos <= index_topk,
    # so this kernel is only ever reached once end_pos exceeds index_topk. By
    # then the cache prefix is already populated, leaving the fallback indexer
    # (which reads k_cache[:, :end_pos]) as the only correct path here.
    return fallback(indexer, x, qr, start_pos, freqs_cis, mask)
