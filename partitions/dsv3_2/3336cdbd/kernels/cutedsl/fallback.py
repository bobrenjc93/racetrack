from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False


_CAT_CACHE = {}


def _cached_cat(*tensors: torch.Tensor | None) -> torch.Tensor | None:
    if any(tensor is None for tensor in tensors):
        return None
    if torch.is_grad_enabled():
        return torch.cat(tensors, dim=0)
    key = tuple(
        (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
            getattr(tensor, "_version", 0),
        )
        for tensor in tensors
    )
    cached = _CAT_CACHE.get(key)
    if cached is None:
        cached = torch.cat(tensors, dim=0).contiguous()
        _CAT_CACHE[key] = cached
    return cached


def fused_norm_rope(
    q_c,
    q_weight,
    kv_c,
    kv_weight,
    k_pe,
    positions,
    *,
    eps,
    rope_base,
    fallback,
):
    return fallback(
        q_c, q_weight, kv_c, kv_weight, k_pe, positions,
        eps=eps, rope_base=rope_base,
    )


def fused_residual_norm(
    residual,
    update,
    norm_weight,
    *,
    eps,
    fallback,
):
    return fallback(residual, update, norm_weight, eps=eps)


def fused_swiglu(gate, up, *, fallback):
    return fallback(gate, up)


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

    from inference import model as real_model

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


def fused_mlp_gate_up_proj(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w3_weight: torch.Tensor,
    w3_scale: torch.Tensor | None,
    *,
    scale_fmt: str | None,
    fallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    del fallback
    gate_features = w1_weight.shape[0]
    if w1_weight.dtype != torch.float8_e4m3fn:
        gate_up = F.linear(x, _cached_cat(w1_weight, w3_weight))
    else:
        from inference import model as real_model

        x_fp8, x_scale = real_model.act_quant(x, real_model.block_size, scale_fmt)
        gate_up = real_model.fp8_gemm(
            x_fp8,
            x_scale,
            _cached_cat(w1_weight, w3_weight),
            _cached_cat(w1_scale, w3_scale),
        )
    return torch.split(gate_up, [gate_features, w3_weight.shape[0]], dim=-1)


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


def hc_head(
    hidden_states,
    hc_fn,
    hc_scale,
    hc_base,
    *,
    rms_norm_eps,
    hc_eps,
    fallback,
):
    return fallback(
        hidden_states, hc_fn, hc_scale, hc_base,
        rms_norm_eps=rms_norm_eps, hc_eps=hc_eps,
    )
