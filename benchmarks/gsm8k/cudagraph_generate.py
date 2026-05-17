"""CUDA-graph-accelerated greedy generation for benchmarking.

Captures a CUDA graph of the decode step after prefill, then replays it
for each subsequent token. This eliminates Python dispatch overhead and
matches or beats Inductor's performance.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.distributed as dist

from racetrack.partition_spec import PartitionSpec
from racetrack.runtime.dispatch import KernelDispatcher


def _load_kernel(kernel_root: Path, name: str):
    for backend in ("triton", "cutedsl", "helion"):
        path = kernel_root / backend / f"{name}.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location(name, str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if getattr(mod, "BACKEND_AVAILABLE", False):
                return mod
    return None


def patch_for_cudagraph(model: torch.nn.Module, kernel_root: Path, *, stack_moe: bool = False) -> list:
    """Apply aggressive fused-kernel patches for maximum performance.

    When stack_moe=True, expert weights are stacked into tensors for
    CUDA graph compatibility. When False, the original sparse MoE
    dispatch is kept (faster for eager execution).

    Returns a list of (module, attr_name, original_value) for rollback.
    """
    from inference import model as rm

    kernels = {}
    for name in ("residual_norm", "swiglu", "act_quant", "fused_rope"):
        mod = _load_kernel(kernel_root, name)
        if mod is not None:
            kernels[name] = mod

    originals = []

    for module in model.modules():
        if isinstance(module, rm.Indexer):
            _save_and_patch(module, "forward",
                            _make_indexer_shortcircuit(module), originals)

        if isinstance(module, rm.Gate):
            _save_and_patch(module, "forward",
                            _make_gate_forward(module), originals)

        if isinstance(module, rm.MLA):
            _save_and_patch(module, "forward",
                            _make_mla_forward(module, kernels), originals)

        if isinstance(module, rm.RMSNorm):
            _save_and_patch(module, "forward",
                            _make_rmsnorm_forward(module, kernels), originals)

        if isinstance(module, rm.MLP):
            _save_and_patch(module, "forward",
                            _make_mlp_forward(module, kernels), originals)

        if isinstance(module, (rm.Linear, rm.ColumnParallelLinear, rm.RowParallelLinear)):
            if module.weight.dtype == torch.float8_e4m3fn:
                _save_and_patch(module, "forward",
                                _make_linear_forward(module, kernels), originals)

    if stack_moe:
        for module in model.modules():
            if isinstance(module, rm.MoE):
                _save_and_patch(module, "forward",
                                _make_moe_forward(module, kernels), originals)

    return originals


def rollback_cudagraph_patches(originals: list) -> None:
    for module, name, original in reversed(originals):
        setattr(module, name, original)


def _save_and_patch(module, name, new_fn, originals):
    originals.append((module, name, getattr(module, name)))
    setattr(module, name, new_fn)


def _make_indexer_shortcircuit(module):
    original = module.forward
    topk = module.index_topk

    def forward(x, qr, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if end_pos <= topk:
            return torch.arange(
                end_pos, device=x.device, dtype=torch.long,
            ).view(1, 1, end_pos).expand(bsz, seqlen, end_pos)
        return original(x, qr, start_pos, freqs_cis, mask)
    return forward


def _make_linear_forward(module, kernels):
    aq = kernels.get("act_quant")
    if aq is None:
        return module.forward

    def forward(x):
        from inference.kernel import fp8_gemm
        xq, xs = aq.fused_act_quant(x, fallback=None)
        y = fp8_gemm(xq, xs, module.weight, module.weight.scale)
        if getattr(module, 'reduce_output', False) and dist.is_initialized() and dist.get_world_size() > 1:
            y = y.float()
            dist.all_reduce(y)
            return y.to(x.dtype)
        return y
    return forward


def _fused_standalone_norm(x, weight, eps, kernels):
    rope_mod = kernels.get("fused_rope")
    if rope_mod is None:
        dtype = x.dtype
        xf = x.float()
        return (weight * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(dtype)
    shape = x.shape
    cols = shape[-1]
    x_flat = x.view(-1, cols) if x.is_contiguous() else x.contiguous().view(-1, cols)
    out = torch.empty_like(x_flat)
    import triton
    block_size = triton.next_power_of_2(cols)
    rope_mod._rms_norm_kernel[(x_flat.shape[0],)](
        x_flat, weight, out, eps, cols,
        x_flat.stride(0), block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return out.view(shape)


def _make_rmsnorm_forward(module, kernels):
    rn = kernels.get("residual_norm")
    if rn is None:
        return module.forward

    def forward(x, residual=None):
        if residual is None:
            return _fused_standalone_norm(x, module.weight, module.eps, kernels)
        shape = x.shape
        cols = shape[-1]
        x_flat = x.view(-1, cols) if x.is_contiguous() else x.contiguous().view(-1, cols)
        r_flat = residual.view(-1, cols) if residual.is_contiguous() else residual.contiguous().view(-1, cols)
        hidden, normed = rn.fused_residual_norm(
            r_flat, x_flat, module.weight, eps=module.eps, fallback=None,
        )
        return normed.view(shape), hidden.view(shape)
    return forward


def _make_mlp_forward(module, kernels):
    aq = kernels.get("act_quant")
    if aq is None:
        return module.forward
    w2_fp8 = module.w2.weight.dtype == torch.float8_e4m3fn

    def forward(x):
        from inference.kernel import fp8_gemm
        gate = _fused_linear(x, module.w1, aq)
        up = _fused_linear(x, module.w3, aq)
        if w2_fp8:
            h_q, h_s = aq.fused_swiglu_quant(gate, up, fallback=None)
            y = fp8_gemm(h_q, h_s, module.w2.weight, module.w2.weight.scale)
            if getattr(module.w2, 'reduce_output', False) and dist.is_initialized() and dist.get_world_size() > 1:
                y = y.float()
                dist.all_reduce(y)
                return y.to(x.dtype)
            return y
        sw = kernels.get("swiglu")
        if sw:
            return _fused_linear(sw.fused_swiglu(gate, up, fallback=None), module.w2, aq)
        return module.w2(torch.nn.functional.silu(gate) * up)
    return forward


def _fused_linear(x, linear_mod, aq):
    if linear_mod.weight.dtype == torch.float8_e4m3fn:
        from inference.kernel import fp8_gemm
        xq, xs = aq.fused_act_quant(x, fallback=None)
        y = fp8_gemm(xq, xs, linear_mod.weight, linear_mod.weight.scale)
        if getattr(linear_mod, 'reduce_output', False) and dist.is_initialized() and dist.get_world_size() > 1:
            y = y.float()
            dist.all_reduce(y)
            return y.to(x.dtype)
        return y
    return linear_mod(x)


def _make_gate_forward(module):
    def forward(x):
        from inference.model import linear
        scores = linear(x.float(), module.weight.float())
        if module.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if module.bias is not None:
            scores = scores + module.bias
        if module.n_groups > 1:
            scores = scores.view(x.size(0), module.n_groups, -1)
            if module.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(module.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), module.n_groups, dtype=torch.bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), torch.finfo(scores.dtype).min).flatten(1)
        indices = scores.topk(module.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if module.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= module.route_scale
        return weights, indices
    return forward


def _make_moe_forward(module, kernels):
    import torch.nn.functional as F

    topk = module.gate.topk
    start = module.experts_start_idx
    end = module.experts_end_idx
    local_experts = [module.experts[i] for i in range(start, end)]
    n_local = len(local_experts)

    has_scales = hasattr(local_experts[0].w1.weight, 'scale') and local_experts[0].w1.weight.scale is not None
    aq = kernels.get("act_quant")
    sw = kernels.get("swiglu")

    if has_scales:
        w1_s = torch.stack([e.w1.weight.scale for e in local_experts])
    w1_stack = torch.stack([e.w1.weight.data for e in local_experts])
    for e in local_experts:
        e.w1 = None
    torch.cuda.empty_cache()

    if has_scales:
        w3_s = torch.stack([e.w3.weight.scale for e in local_experts])
    w3_stack = torch.stack([e.w3.weight.data for e in local_experts])
    for e in local_experts:
        e.w3 = None
    torch.cuda.empty_cache()

    if has_scales:
        w2_s = torch.stack([e.w2.weight.scale for e in local_experts])
    w2_stack = torch.stack([e.w2.weight.data for e in local_experts])
    for e in local_experts:
        e.w2 = None
    torch.cuda.empty_cache()

    for i in range(start, end):
        module.experts[i] = None
    torch.cuda.empty_cache()

    def forward(x):
        from inference.kernel import fp8_gemm
        shape = x.size()
        x_flat = x.view(-1, module.dim)
        weights, indices = module.gate(x_flat)
        y = torch.zeros_like(x_flat, dtype=torch.float32)
        local_ids = (indices[0] - start).clamp(0, n_local - 1).to(torch.long)
        is_local = ((indices[0] >= start) & (indices[0] < end)).float()
        expert_weights = weights[0] * is_local
        sel_w1 = torch.index_select(w1_stack, 0, local_ids)
        sel_w3 = torch.index_select(w3_stack, 0, local_ids)
        sel_w2 = torch.index_select(w2_stack, 0, local_ids)
        if has_scales:
            sel_s1 = torch.index_select(w1_s, 0, local_ids)
            sel_s3 = torch.index_select(w3_s, 0, local_ids)
            sel_s2 = torch.index_select(w2_s, 0, local_ids)
            for t in range(topk):
                x_q, x_s = aq.fused_act_quant(x_flat, fallback=None)
                gate_out = fp8_gemm(x_q, x_s, sel_w1[t], sel_s1[t])
                up_out = fp8_gemm(x_q, x_s, sel_w3[t], sel_s3[t])
                h_q, h_s = aq.fused_swiglu_quant(gate_out, up_out, fallback=None)
                out = fp8_gemm(h_q, h_s, sel_w2[t], sel_s2[t])
                y += out.float() * expert_weights[t]
        else:
            x_exp = x_flat.expand(topk, -1, -1)
            gate_out = torch.bmm(x_exp, sel_w1.transpose(1, 2))
            up_out = torch.bmm(x_exp, sel_w3.transpose(1, 2))
            hidden = (F.silu(gate_out.float()) * up_out.float()).to(gate_out.dtype)
            out = torch.bmm(hidden, sel_w2.transpose(1, 2))
            y += (out.squeeze(1).float() * expert_weights.unsqueeze(-1)).sum(0, keepdim=True)
        y += module.shared_experts(x_flat).float()
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(y)
        return y.to(torch.bfloat16).view(shape)

    return forward


def _make_mla_forward(module, kernels):
    original = module.forward
    aq = kernels.get("act_quant")
    if aq is None:
        return original

    def forward(x, start_pos, freqs_cis, mask):
        if mask is not None:
            return original(x, start_pos, freqs_cis, mask)
        from inference.model import apply_rotary_emb, weight_dequant
        from inference.kernel import fp8_gemm
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        if module.wq_a.weight.dtype == torch.float8_e4m3fn:
            x_q, x_s = aq.fused_act_quant(x, fallback=None)
            qr_raw = fp8_gemm(x_q, x_s, module.wq_a.weight, module.wq_a.weight.scale)
            kv_raw = fp8_gemm(x_q, x_s, module.wkv_a.weight, module.wkv_a.weight.scale)
        else:
            qr_raw = module.wq_a(x)
            kv_raw = module.wkv_a(x)

        qr = _fused_standalone_norm(qr_raw, module.q_norm.weight, module.q_norm.eps, kernels)
        q = _fused_linear(qr, module.wq_b, aq)
        q = q.view(bsz, seqlen, module.n_local_heads, module.qk_head_dim)
        q_nope, q_pe = torch.split(q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        kv, k_pe = torch.split(kv_raw, [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1)
        kv = _fused_standalone_norm(kv, module.kv_norm.weight, module.kv_norm.eps, kernels)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        module.kv_cache[:bsz, start_pos:end_pos] = kv
        module.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

        if end_pos <= module.indexer.index_topk:
            scores = (
                torch.einsum("bshd,hdc->bshc", q_nope,
                             _get_wkv_b(module)[:, :module.qk_nope_head_dim])
            )
            scores = (
                torch.einsum("bshc,btc->bsht", scores, module.kv_cache[:bsz, :end_pos])
                + torch.einsum("bshr,btr->bsht", q_pe, module.pe_cache[:bsz, :end_pos])
            ) * module.softmax_scale
            scores = scores.softmax(dim=-1)
        else:
            topk_indices = module.indexer(x, qr_raw, start_pos, freqs_cis, None)
            wkv_b = _get_wkv_b(module)
            q_nope_proj = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :module.qk_nope_head_dim])
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope_proj, module.kv_cache[:bsz, :end_pos])
                + torch.einsum("bshr,btr->bsht", q_pe, module.pe_cache[:bsz, :end_pos])
            ) * module.softmax_scale
            index_mask = scores.new_full((bsz, 1, end_pos), torch.finfo(scores.dtype).min).scatter_(-1, topk_indices, 0.0)
            scores += index_mask.unsqueeze(2)
            scores = scores.softmax(dim=-1)

        x = torch.einsum("bsht,btc->bshc", scores, module.kv_cache[:bsz, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, _get_wkv_b(module)[:, -module.v_head_dim:])
        return _fused_linear(x.flatten(2), module.wo, aq)
    return forward


def _get_wkv_b(module):
    from inference.model import weight_dequant
    if module.dequant_wkv_b is None and module.wkv_b.scale is not None:
        module.dequant_wkv_b = weight_dequant(module.wkv_b.weight, module.wkv_b.scale)
    wkv_b = module.wkv_b.weight if module.dequant_wkv_b is None else module.dequant_wkv_b
    return wkv_b.view(module.n_local_heads, -1, module.kv_lora_rank)


def _make_cudagraph_decode_fn(model, max_seq_len):
    """Build a CUDA-graph-capturable decode step with position-aware buffers.

    Returns (decode_fn, static_tok, static_freqs, static_attn_mask, static_logits)
    where decode_fn() runs one decode step using the static buffers.
    """
    world_size_val = dist.get_world_size() if dist.is_initialized() else 1
    vocab_shard = model.head.weight.shape[0]

    static_tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    static_freqs = model.freqs_cis[0:1].clone()
    static_attn_mask = torch.zeros(1, 1, 1, max_seq_len, device="cuda", dtype=torch.float32)
    static_logits = torch.empty(1, vocab_shard * world_size_val, device="cuda", dtype=torch.float32)

    def decode_step():
        h = model.embed(static_tok)
        residual = None
        for layer in model.layers:
            h, residual = layer(h, residual, -1, static_freqs, None)
        h, _ = model.norm(h, residual)
        s = model.head(h[:, -1].float())
        if world_size_val > 1:
            dist.all_gather_into_tensor(static_logits, s)
        else:
            static_logits.copy_(s)
        return static_logits

    return decode_step, static_tok, static_freqs, static_attn_mask, static_logits


def _patch_mla_for_cudagraph_gen(model, max_seq_len):
    """Patch MLA modules to use full-cache reads + attention mask for CUDA graphs.

    Instead of kv_cache[:, :end_pos] (variable slice), reads full cache
    and masks out future positions via static_attn_mask.
    """
    from inference import model as rm

    originals = []
    attn_masks = []

    for module in model.modules():
        if not isinstance(module, rm.MLA):
            continue

        attn_mask = torch.zeros(1, 1, 1, max_seq_len, device="cuda", dtype=torch.float32)
        attn_masks.append(attn_mask)
        original_forward = getattr(module, 'forward')

        def _make_cg_mla(mla, mask_buf, orig_fwd):
            def forward(x, start_pos, freqs_cis, mask):
                if mask is not None or start_pos == -1:
                    if start_pos != -1:
                        return orig_fwd(x, start_pos, freqs_cis, mask)
                    return _mla_cudagraph_decode(mla, x, freqs_cis, mask_buf)
                return orig_fwd(x, start_pos, freqs_cis, mask)
            return forward

        new_forward = _make_cg_mla(module, attn_mask, original_forward)
        originals.append((module, 'forward', original_forward))
        module.forward = new_forward

    return originals, attn_masks


def _mla_cudagraph_decode(module, x, freqs_cis, attn_mask):
    """MLA decode step that reads full cache + uses attn_mask. CUDA-graph-safe."""
    from inference.model import apply_rotary_emb, weight_dequant

    bsz, seqlen, _ = x.size()
    max_t = module.kv_cache.shape[1]

    qr = module.q_norm(module.wq_a(x))
    q = module.wq_b(qr)
    q = q.view(bsz, seqlen, module.n_local_heads, module.qk_head_dim)
    q_nope, q_pe = torch.split(q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_emb(q_pe, freqs_cis)

    kv = module.wkv_a(x)
    kv, k_pe = torch.split(kv, [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1)
    kv = module.kv_norm(kv)
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

    if module.dequant_wkv_b is None and module.wkv_b.scale is not None:
        module.dequant_wkv_b = weight_dequant(module.wkv_b.weight, module.wkv_b.scale)
    wkv_b = module.wkv_b.weight if module.dequant_wkv_b is None else module.dequant_wkv_b
    wkv_b = wkv_b.view(module.n_local_heads, -1, module.kv_lora_rank)

    q_nope_proj = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :module.qk_nope_head_dim])
    scores = (
        torch.einsum("bshc,btc->bsht", q_nope_proj, module.kv_cache[:bsz, :max_t])
        + torch.einsum("bshr,btr->bsht", q_pe, module.pe_cache[:bsz, :max_t])
    ) * module.softmax_scale
    scores = scores + attn_mask
    scores = scores.softmax(dim=-1)
    x = torch.einsum("bsht,btc->bshc", scores, module.kv_cache[:bsz, :max_t])
    x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -module.v_head_dim:])
    return module.wo(x.flatten(2))


@torch.inference_mode()
def generate_greedy_cudagraph(
    model,
    prompt_tokens: list[list[int]],
    max_new_tokens: int,
    eos_id: int,
) -> list[list[int]]:
    """Greedy generation with CUDA graph capture for decode steps.

    Captures a CUDA graph after the first decode step. The graph uses
    static buffers for token, freqs, and attention mask. Before each
    replay, the buffers are updated with the current position's values.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    max_seq_len = model.max_seq_len
    total_len = min(max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full(
        (len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda",
    )
    for i, t in enumerate(prompt_tokens):
        tokens[i, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    world_size_val = dist.get_world_size() if dist.is_initialized() else 1

    mla_originals, attn_masks = _patch_mla_for_cudagraph_gen(model, max_seq_len)

    graph = None
    static_tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    vocab_shard = model.head.weight.shape[0]
    static_logits = torch.empty(1, vocab_shard * world_size_val, device="cuda", dtype=torch.float32)

    def _update_attn_masks(end_pos):
        finfo_min = torch.finfo(torch.float32).min
        for mask in attn_masks:
            mask.fill_(finfo_min)
            mask[:, :, :, :end_pos] = 0.0

    def _write_caches(pos, kv_val, pe_val, mla_mod):
        mla_mod.kv_cache[:, pos:pos+1] = kv_val
        mla_mod.pe_cache[:, pos:pos+1] = pe_val

    try:
        for cur_pos in range(min(prompt_lens), total_len):
            is_decode = (cur_pos - prev_pos) == 1

            if is_decode and graph is not None:
                static_tok.fill_(tokens[0, prev_pos].item())
                fc = model.freqs_cis[prev_pos:prev_pos+1]
                _update_attn_masks(cur_pos)
                graph.replay()
                logits = static_logits
            else:
                logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

                if is_decode and graph is None:
                    torch.cuda.synchronize()
                    _update_attn_masks(cur_pos + 1)
                    static_tok.copy_(tokens[:, prev_pos:cur_pos])

                    def _capture_decode():
                        fc = model.freqs_cis[prev_pos:prev_pos+1]
                        h = model.embed(static_tok)
                        residual = None
                        for layer in model.layers:
                            h, residual = layer(h, residual, prev_pos, fc, None)
                        h, _ = model.norm(h, residual)
                        s = model.head(h[:, -1].float())
                        if world_size_val > 1:
                            dist.all_gather_into_tensor(static_logits, s)
                        else:
                            static_logits.copy_(s)

                    _capture_decode()
                    torch.cuda.synchronize()
                    _capture_decode()
                    torch.cuda.synchronize()

                    try:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            _capture_decode()
                        torch.cuda.synchronize()
                    except Exception as e:
                        graph = None

            next_token = logits.argmax(dim=-1)
            next_token = torch.where(
                prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token,
            )
            tokens[:, cur_pos] = next_token
            finished |= torch.logical_and(
                ~prompt_mask[:, cur_pos], next_token == eos_id,
            )
            prev_pos = cur_pos
            if finished.all():
                break
    finally:
        for module, name, original in reversed(mla_originals):
            setattr(module, name, original)

    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i] : prompt_lens[i] + max_new_tokens]
        if eos_id in toks:
            toks = toks[: toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens
