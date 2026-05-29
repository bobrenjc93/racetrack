"""Pre-trace model patchers: modify module.forward before Dynamo traces.

Each patcher is registered by op name and modifies specific module types
to use fused kernel implementations. The patched model is then traced by
torch.compile, so the FX graph already reflects the optimized code paths.

Pre-trace patches handle control flow changes (pre_trace kind) and whole
module forward replacements (module_patch kind) that cannot be expressed
as FX subgraph pattern matches.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.distributed as dist

from racetrack.partition_spec import PartitionSpec
from racetrack.runtime.dispatch import KernelDispatcher


_REGISTRY: dict[str, Callable] = {}

Originals = list[tuple[nn.Module, str, bool, Any]]


def register_patcher(name: str):
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return decorator


def apply_pre_trace_patches(
    model: nn.Module,
    spec: PartitionSpec,
    dispatcher: KernelDispatcher,
) -> Originals:
    originals: Originals = []
    for op in spec.pre_trace_ops + spec.module_patch_ops:
        patcher = _REGISTRY.get(op.name)
        if patcher is None:
            continue
        patcher(model, dispatcher, originals)
    return originals


def rollback_patches(originals: Originals) -> None:
    for module, name, existed, original in reversed(originals):
        if existed:
            setattr(module, name, original)
        else:
            delattr(module, name)


def _set_attr(
    module: nn.Module,
    name: str,
    value: Any,
    originals: Originals,
) -> None:
    existed = hasattr(module, name)
    original = getattr(module, name, None)
    originals.append((module, name, existed, original))
    setattr(module, name, value)


def _resolve_kernel(dispatcher: KernelDispatcher, op_name: str):
    backend = dispatcher.selected_backend(default="torch")
    if backend == "best":
        backend = dispatcher._best_fast_path.get(op_name, "triton")
    if backend == "torch" or backend == "all":
        return None
    return dispatcher._resolve(backend, op_name)


# ---------------------------------------------------------------------------
# Patchers: dsv3_2 + dsv3_2_nvfp4 shared ops
# ---------------------------------------------------------------------------


def _write_indexer_k_cache(module, x, start_pos, freqs_cis) -> None:
    """Populate the indexer K-cache for positions short-circuited by full-topk.

    When end_pos <= index_topk the indexer returns arange instead of running
    the real forward, so its k_cache/k_scale_cache buffers (which the fallback
    indexer later reads over the whole [0, end_pos) prefix) would stay stale.
    This mirrors the K-cache write in Indexer.forward so the cached keys for
    short-circuited positions are correct once the fallback path eventually runs.
    """
    # Lightweight indexer stubs (and any variant without the K-cache buffers)
    # have nothing to populate; skip so the short-circuit stays a pure arange
    # return for callers that do not maintain a K-cache of their own.
    if not hasattr(module, "k_cache"):
        return

    from racetrack.models import deepseek as rm

    bsz, seqlen, _ = x.size()
    end_pos = start_pos + seqlen
    k = module.wk(x)
    k = module.k_norm(k)
    k_pe, k_nope = torch.split(
        k, [module.rope_head_dim, module.head_dim - module.rope_head_dim], dim=-1,
    )
    k_pe = rm.apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)
    k = rm.rotate_activation(k)
    k_fp8, k_scale = rm.act_quant(k, rm.block_size, module.scale_fmt)
    module.k_cache[:bsz, start_pos:end_pos] = k_fp8
    module.k_scale_cache[:bsz, start_pos:end_pos] = k_scale


@register_patcher("fused_full_topk_indexer")
def _patch_full_topk_indexer(
    model: nn.Module,
    dispatcher: KernelDispatcher,
    originals: Originals,
) -> None:
    from racetrack.models import deepseek as rm

    kernel_fn = _resolve_kernel(dispatcher, "fused_full_topk_indexer")

    for module in model.modules():
        if not isinstance(module, rm.Indexer):
            continue
        original_forward = module.forward
        topk = module.index_topk

        if kernel_fn is not None:
            def _make_forward(orig, top_k, kfn, indexer):
                def forward(x, qr, start_pos, freqs_cis, mask):
                    bsz, seqlen, _ = x.size()
                    end_pos = start_pos + seqlen
                    if end_pos <= top_k:
                        _write_indexer_k_cache(indexer, x, start_pos, freqs_cis)
                        return torch.arange(
                            end_pos, device=x.device, dtype=torch.long,
                        ).view(1, 1, end_pos).expand(bsz, seqlen, end_pos)
                    def fallback(indexer, hidden, q_residual, pos, freqs, attn_mask):
                        del indexer
                        return orig(hidden, q_residual, pos, freqs, attn_mask)
                    return kfn(indexer, x, qr, start_pos, freqs_cis, mask, fallback=fallback)
                return forward
            _set_attr(module, "forward", _make_forward(original_forward, topk, kernel_fn, module), originals)
        else:
            def _make_forward_shortcircuit(orig, top_k, indexer):
                def forward(x, qr, start_pos, freqs_cis, mask):
                    bsz, seqlen, _ = x.size()
                    end_pos = start_pos + seqlen
                    if end_pos <= top_k:
                        _write_indexer_k_cache(indexer, x, start_pos, freqs_cis)
                        return torch.arange(
                            end_pos, device=x.device, dtype=torch.long,
                        ).view(1, 1, end_pos).expand(bsz, seqlen, end_pos)
                    return orig(x, qr, start_pos, freqs_cis, mask)
                return forward
            _set_attr(module, "forward", _make_forward_shortcircuit(original_forward, topk, module), originals)

    # MLA full-topk shortcircuit is handled by the indexer patch above.
    # Skipping MLA forward patching to avoid conflicts with other patchers.


def _mla_full_topk_forward(module, x, start_pos, freqs_cis, mask):
    from racetrack.models import deepseek as real_model

    bsz, seqlen, _ = x.size()
    end_pos = start_pos + seqlen
    qr = module.q_norm(module.wq_a(x))
    q = module.wq_b(qr)
    q = q.view(bsz, seqlen, module.n_local_heads, module.qk_head_dim)
    q_nope, q_pe = torch.split(q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1)
    q_pe = real_model.apply_rotary_emb(q_pe, freqs_cis)
    kv = module.wkv_a(x)
    kv, k_pe = torch.split(kv, [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1)
    kv = module.kv_norm(kv)
    k_pe = real_model.apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
    module.kv_cache[:bsz, start_pos:end_pos] = kv
    module.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
    module.indexer(x, qr, start_pos, freqs_cis, mask)

    if mask is not None:
        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = module.wkv_b(kv)
        kv = kv.view(bsz, seqlen, module.n_local_heads, module.qk_nope_head_dim + module.v_head_dim)
        k_nope, v = torch.split(kv, [module.qk_nope_head_dim, module.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, module.n_local_heads, -1)], dim=-1)
        scores = torch.einsum("bshd,bthd->bsht", q, k).mul_(module.softmax_scale)
        scores += mask.unsqueeze(0).unsqueeze(2)
        scores = scores.softmax(dim=-1)
        x = torch.einsum("bsht,bthd->bshd", scores, v)
    else:
        if module.dequant_wkv_b is None and module.wkv_b.scale is not None:
            module.dequant_wkv_b = real_model.weight_dequant(module.wkv_b.weight, module.wkv_b.scale)
        wkv_b = module.wkv_b.weight if module.dequant_wkv_b is None else module.dequant_wkv_b
        wkv_b = wkv_b.view(module.n_local_heads, -1, module.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :module.qk_nope_head_dim])
        scores = (
            torch.einsum("bshc,btc->bsht", q_nope, module.kv_cache[:bsz, :end_pos])
            + torch.einsum("bshr,btr->bsht", q_pe, module.pe_cache[:bsz, :end_pos])
        ) * module.softmax_scale
        scores = scores.softmax(dim=-1)
        x = torch.einsum("bsht,btc->bshc", scores, module.kv_cache[:bsz, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -module.v_head_dim:])
    return module.wo(x.flatten(2))


@register_patcher("fused_single_token_moe")
def _patch_single_token_moe(
    model: nn.Module,
    dispatcher: KernelDispatcher,
    originals: Originals,
) -> None:
    from racetrack.models import deepseek as rm

    kernel_fn = _resolve_kernel(dispatcher, "fused_single_token_moe")
    if kernel_fn is None:
        return

    for module in model.modules():
        if not isinstance(module, rm.MoE):
            continue
        original_forward = module.forward

        def _make_forward(orig, moe_mod, kfn):
            def forward(x):
                def fallback(moe, hidden):
                    del moe
                    return orig(hidden)
                return kfn(moe_mod, x, fallback=fallback)
            return forward
        _set_attr(module, "forward", _make_forward(original_forward, module, kernel_fn), originals)


@register_patcher("fused_mlp_gate_up_proj")
def _patch_mlp_gate_up_proj(
    model: nn.Module,
    dispatcher: KernelDispatcher,
    originals: Originals,
) -> None:
    from racetrack.models import deepseek as rm

    gate_up_fn = _resolve_kernel(dispatcher, "fused_mlp_gate_up_proj")
    swiglu_fn = _resolve_kernel(dispatcher, "fused_swiglu")
    if gate_up_fn is None:
        return

    for module in model.modules():
        if not isinstance(module, rm.MLP):
            continue

        def _make_forward(mlp_mod, gu_fn, sw_fn):
            def forward(x):
                def gate_up_fallback(hidden, w1_w, w1_s, w3_w, w3_s, *, scale_fmt):
                    del w1_s, w3_s
                    return (
                        rm.linear(hidden, w1_w, None, scale_fmt),
                        rm.linear(hidden, w3_w, None, scale_fmt),
                    )
                gate, up = gu_fn(
                    x, mlp_mod.w1.weight, mlp_mod.w1.scale,
                    mlp_mod.w3.weight, mlp_mod.w3.scale,
                    scale_fmt=mlp_mod.w1.scale_fmt, fallback=gate_up_fallback,
                )
                if sw_fn is not None:
                    def swiglu_fallback(g, u):
                        return (torch.nn.functional.silu(g.float()) * u.float()).type_as(g)
                    hidden = sw_fn(gate, up, fallback=swiglu_fallback)
                else:
                    hidden = (torch.nn.functional.silu(gate.float()) * up.float()).type_as(gate)
                return mlp_mod.w2(hidden.type_as(x))
            return forward
        _set_attr(module, "forward", _make_forward(module, gate_up_fn, swiglu_fn), originals)


# ---------------------------------------------------------------------------
# Patchers: dsv3_2 deep-fusion ops (d299a5e3, eba37e0d, 73a281a9)
# ---------------------------------------------------------------------------


@register_patcher("fused_attn_norm_qkv")
def _patch_attn_norm_qkv(
    model: nn.Module,
    dispatcher: KernelDispatcher,
    originals: Originals,
) -> None:
    kernel_fn = _resolve_kernel(dispatcher, "fused_attn_norm_qkv")
    if kernel_fn is None:
        return
    from racetrack.models import deepseek as rm
    for module in model.modules():
        if isinstance(module, (rm.RMSNorm, rm.MLA)):
            _set_attr(module, "kernel_dispatcher", dispatcher, originals)
            _set_attr(module, "kernel_stats", None, originals)


@register_patcher("fused_qkv_proj_rope")
def _patch_qkv_proj_rope(
    model: nn.Module,
    dispatcher: KernelDispatcher,
    originals: Originals,
) -> None:
    kernel_fn = _resolve_kernel(dispatcher, "fused_qkv_proj_rope")
    if kernel_fn is None:
        return
    from racetrack.models import deepseek as rm
    for module in model.modules():
        if not isinstance(module, rm.MLA):
            continue
        _set_attr(module, "kernel_dispatcher", dispatcher, originals)
        _set_attr(module, "kernel_stats", None, originals)


# ---------------------------------------------------------------------------
# Patchers: dsv3_2_nvfp4 ops
# ---------------------------------------------------------------------------


@register_patcher("fused_ar_rms_qkv_proj")
def _patch_ar_rms_qkv_proj(
    model: nn.Module,
    dispatcher: KernelDispatcher,
    originals: Originals,
) -> None:
    kernel_fn = _resolve_kernel(dispatcher, "fused_ar_rms_qkv_proj")
    if kernel_fn is None:
        return
    from racetrack.models import deepseek as rm
    for module in model.modules():
        if isinstance(module, (rm.RMSNorm, rm.MLA)):
            _set_attr(module, "kernel_dispatcher", dispatcher, originals)
            _set_attr(module, "kernel_stats", None, originals)


@register_patcher("fused_indexer_k_path")
def _patch_indexer_k_path(
    model: nn.Module,
    dispatcher: KernelDispatcher,
    originals: Originals,
) -> None:
    kernel_fn = _resolve_kernel(dispatcher, "fused_indexer_k_path")
    if kernel_fn is None:
        return
    from racetrack.models import deepseek as rm
    for module in model.modules():
        if isinstance(module, rm.Indexer):
            _set_attr(module, "kernel_dispatcher", dispatcher, originals)
            _set_attr(module, "kernel_stats", None, originals)


@register_patcher("fused_q_indexer_score")
def _patch_q_indexer_score(
    model: nn.Module,
    dispatcher: KernelDispatcher,
    originals: Originals,
) -> None:
    kernel_fn = _resolve_kernel(dispatcher, "fused_q_indexer_score")
    if kernel_fn is None:
        return
    from racetrack.models import deepseek as rm
    for module in model.modules():
        if isinstance(module, rm.MLA):
            _set_attr(module, "kernel_dispatcher", dispatcher, originals)
            _set_attr(module, "kernel_stats", None, originals)


@register_patcher("fused_q_rope_quant")
def _patch_q_rope_quant(
    model: nn.Module,
    dispatcher: KernelDispatcher,
    originals: Originals,
) -> None:
    kernel_fn = _resolve_kernel(dispatcher, "fused_q_rope_quant")
    if kernel_fn is None:
        return
    from racetrack.models import deepseek as rm
    for module in model.modules():
        if isinstance(module, rm.MLA):
            _set_attr(module, "kernel_dispatcher", dispatcher, originals)
            _set_attr(module, "kernel_stats", None, originals)


@register_patcher("fused_prefill_qkv_b_rope_cache")
def _patch_prefill_qkv_b_rope_cache(
    model: nn.Module,
    dispatcher: KernelDispatcher,
    originals: Originals,
) -> None:
    kernel_fn = _resolve_kernel(dispatcher, "fused_prefill_qkv_b_rope_cache")
    if kernel_fn is None:
        return
    from racetrack.models import deepseek as rm
    for module in model.modules():
        if isinstance(module, rm.MLA):
            _set_attr(module, "kernel_dispatcher", dispatcher, originals)
            _set_attr(module, "kernel_stats", None, originals)


# ---------------------------------------------------------------------------
# Patchers: dsv3_2_nvfp4 prefill ops
# ---------------------------------------------------------------------------


@register_patcher("fused_attn_norm_qkv_prefill")
def _patch_attn_norm_qkv_prefill(
    model: nn.Module,
    dispatcher: KernelDispatcher,
    originals: Originals,
) -> None:
    pass


@register_patcher("fused_norms_rope_cache_prefill")
def _patch_norms_rope_cache_prefill(
    model: nn.Module,
    dispatcher: KernelDispatcher,
    originals: Originals,
) -> None:
    pass


@register_patcher("fused_q_prefill_proj")
def _patch_q_prefill_proj(
    model: nn.Module,
    dispatcher: KernelDispatcher,
    originals: Originals,
) -> None:
    pass


@register_patcher("fused_q_rope_prefill")
def _patch_q_rope_prefill(
    model: nn.Module,
    dispatcher: KernelDispatcher,
    originals: Originals,
) -> None:
    pass
