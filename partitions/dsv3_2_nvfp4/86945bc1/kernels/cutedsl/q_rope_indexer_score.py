from __future__ import annotations

import torch

try:
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401

    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False


_FLOAT_TENSOR_CACHE = {}


def _cached_float_tensor(tensor):
    if tensor.dtype == torch.float32:
        return tensor
    if torch.is_grad_enabled():
        return tensor.float()
    key = (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        getattr(tensor, "_version", 0),
    )
    cached = _FLOAT_TENSOR_CACHE.get(key)
    if cached is None:
        cached = tensor.float().contiguous()
        _FLOAT_TENSOR_CACHE[key] = cached
    return cached


def _hadamard_transform(x, H):
    d = x.shape[-1]
    return (x.float() @ _cached_float_tensor(H)[:d, :d] * (d ** -0.5)).type_as(x)


def _act_quant_unit_scale(x, block_size, act_quant):
    if x.size(-1) % block_size == 0:
        return act_quant(x, block_size)
    try:
        return x.contiguous().to(torch.float8_e4m3fn), None
    except RuntimeError:
        return x.float(), None


def _fp8_index_unit_k_scale(q, q_s, k):
    logits = torch.einsum("bmhd,bnd->bmhn", q.float(), k.float())
    logits = torch.relu(logits) * q_s.unsqueeze(-1)
    return logits.sum(dim=2)


def fused_q_rope_quant(
    q_pe, idx_q, indexer_w,
    freqs_cis, H,
    idx_k_cache, idx_k_scale_cache,
    *, rope_head_dim, idx_n_heads, idx_softmax_scale,
    end_pos, block_size,
    fallback,
):
    del fallback
    import importlib
    _model = importlib.import_module("partitions.dsv3_2_nvfp4.model")
    apply_rotary_emb = _model.apply_rotary_emb
    act_quant = _model.act_quant
    fp8_index = _model.fp8_index
    DSV3_2_NVFP4_CONFIG = _model.DSV3_2_NVFP4_CONFIG
    bsz, seqlen = q_pe.shape[0], q_pe.shape[1]

    q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=True)
    topk_count = min(DSV3_2_NVFP4_CONFIG.index_topk, end_pos)
    if topk_count == end_pos:
        topk_indices = torch.arange(
            end_pos, device=q_pe.device, dtype=torch.long,
        ).view(1, 1, end_pos).expand(bsz, seqlen, end_pos)
        return q_pe, topk_indices

    idx_q_pe, idx_q_nope = torch.split(
        idx_q, [rope_head_dim, idx_q.shape[-1] - rope_head_dim], dim=-1,
    )
    idx_q_pe = apply_rotary_emb(idx_q_pe, freqs_cis, interleaved=False)
    idx_q = torch.cat([idx_q_pe, idx_q_nope], dim=-1)
    idx_q = _hadamard_transform(idx_q, H)
    idx_q_fp8, idx_q_scale = _act_quant_unit_scale(idx_q, block_size, act_quant)

    weights = indexer_w * idx_n_heads ** -0.5
    if idx_q_scale is None:
        weights = weights * idx_softmax_scale
    else:
        weights = (weights.unsqueeze(-1) * idx_q_scale * idx_softmax_scale).squeeze(-1)

    k_cached = idx_k_cache[:bsz, :end_pos].contiguous()
    if idx_k_cache.shape[-1] % block_size == 0:
        k_s = idx_k_scale_cache[:bsz, :end_pos].squeeze(-1).contiguous()
        index_score = fp8_index(idx_q_fp8.float(), weights, k_cached, k_s)
    else:
        index_score = _fp8_index_unit_k_scale(idx_q_fp8.float(), weights, k_cached)
    topk_indices = index_score.topk(topk_count, dim=-1)[1]

    return q_pe, topk_indices
