from __future__ import annotations

import torch

try:
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401

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
    bsz, seqlen, _ = x.size()
    end_pos = start_pos + seqlen
    if start_pos == 0:
        indexer._racetrack_pending_full_topk = []
    elif not hasattr(indexer, "_racetrack_pending_full_topk"):
        indexer._racetrack_pending_full_topk = []
    if end_pos > indexer.index_topk:
        _flush_pending_full_topk(indexer)
        return fallback(indexer, x, qr, start_pos, freqs_cis, mask)

    indexer._racetrack_pending_full_topk.append((start_pos, x, freqs_cis))
    return torch.arange(
        end_pos,
        device=x.device,
        dtype=torch.long,
    ).view(1, 1, end_pos).expand(bsz, seqlen, end_pos).contiguous()


def _flush_pending_full_topk(indexer) -> None:
    pending = getattr(indexer, "_racetrack_pending_full_topk", [])
    if not pending:
        return
    for start_pos, x, freqs_cis in pending:
        _write_indexer_k_cache(indexer, x, start_pos, freqs_cis)
    pending.clear()


def _write_indexer_k_cache(
    indexer,
    x: torch.Tensor,
    start_pos: int,
    freqs_cis: torch.Tensor,
) -> None:
    bsz, seqlen, _ = x.size()
    end_pos = start_pos + seqlen

    from racetrack.models import deepseek as real_model

    k = indexer.wk(x)
    k = indexer.k_norm(k)
    k_pe, k_nope = torch.split(
        k,
        [indexer.rope_head_dim, indexer.head_dim - indexer.rope_head_dim],
        dim=-1,
    )
    k_pe = real_model.apply_rotary_emb(
        k_pe.unsqueeze(2),
        freqs_cis,
        False,
    ).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)
    k = real_model.rotate_activation(k)
    k_fp8, k_scale = real_model.act_quant(
        k,
        real_model.block_size,
        indexer.scale_fmt,
    )
    indexer.k_cache[:bsz, start_pos:end_pos] = k_fp8
    indexer.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
