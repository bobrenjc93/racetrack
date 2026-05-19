"""Flat decode function: Inductor-style Python codegen with custom Triton kernels.

Generates a flat decode function that calls fused kernels directly,
bypassing nn.Module.__call__ overhead. This is exactly what Inductor
does: Python wrapper code + Triton kernels.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.distributed as dist


def _load_kernel(kernel_root: Path, name: str, backend: str | None = None):
    """Load a kernel module by name from the specified backend only."""
    backends = [backend] if backend else ("triton", "cutedsl", "helion")
    for b in backends:
        path = kernel_root / b / f"{name}.py"
        if path.exists():
            spec_name = f"{name}_{b}_{abs(hash(str(path)))}"
            spec = importlib.util.spec_from_file_location(spec_name, str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if getattr(mod, "BACKEND_AVAILABLE", False):
                return mod
    return None


def build_flat_decode(model, kernel_root: Path | None = None, backend: str | None = None):
    """Build a flat decode function from the model + optional partition kernels.

    When kernel_root is None, uses inference.kernel defaults (baseline mode).
    When backend is specified, only loads kernels from that backend directory.
    Returns (flat_decode, flat_decode_cg, update_bufs, static_logits).
    """
    from inference.model import apply_rotary_emb, weight_dequant
    from inference.kernel import fp8_gemm
    from inference import kernel as inf_kernel

    aq = _load_kernel(kernel_root, "act_quant", backend) if kernel_root else None
    rn = _load_kernel(kernel_root, "residual_norm", backend) if kernel_root else None
    fr = _load_kernel(kernel_root, "fused_rope", backend) if kernel_root else None

    ws = dist.get_world_size() if dist.is_initialized() else 1

    embed_weight = model.embed.weight
    embed_vsi = model.embed.vocab_start_idx
    embed_vei = model.embed.vocab_end_idx
    embed_pvs = model.embed.part_vocab_size
    _zero_scalar = torch.tensor(0, device="cuda", dtype=torch.long)
    _zero_embed = torch.zeros(1, model.embed.dim, device="cuda", dtype=embed_weight.dtype)

    def _cg_safe_embed(x):
        if ws > 1:
            mask = (x < embed_vsi) | (x >= embed_vei)
            local_x = (x - embed_vsi).clamp(0, embed_pvs - 1)
            y = torch.nn.functional.embedding(local_x, embed_weight)
            y = y * (~mask).unsqueeze(-1).to(y.dtype)
            dist.all_reduce(y)
            return y
        return torch.nn.functional.embedding(x, embed_weight)

    freqs_cis = model.freqs_cis
    norm_weight = model.norm.weight
    norm_eps = model.norm.eps
    head_weight = model.head.weight
    head_scale = model.head.scale if hasattr(model.head, 'scale') else None

    layers = []
    for layer in model.layers:
        ld = _extract_layer(layer, aq, rn, ws)
        layers.append(ld)

    def _do_act_quant(x):
        if aq is not None:
            return aq.fused_act_quant(x, fallback=None)
        return inf_kernel.act_quant(x)

    def _do_swiglu_quant(gate, up):
        if aq is not None and hasattr(aq, 'fused_swiglu_quant'):
            return aq.fused_swiglu_quant(gate, up, fallback=None)
        h = (torch.nn.functional.silu(gate.float()) * up.float()).to(gate.dtype)
        return _do_act_quant(h)

    def _rms_norm_eager(x, weight, eps):
        dtype = x.dtype
        xf = x.float()
        return (weight * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(dtype)

    def _residual_norm(x, residual, weight, eps):
        if rn is not None:
            shape = x.shape
            cols = shape[-1]
            x_f = x.contiguous().view(-1, cols)
            r_f = residual.contiguous().view(-1, cols)
            hidden, normed = rn.fused_residual_norm(r_f, x_f, weight, eps=eps, fallback=None)
            return normed.view(shape), hidden.view(shape)
        hidden = residual + x
        normed = _rms_norm_eager(hidden, weight, eps)
        return normed, hidden

    def _standalone_norm(x, weight, eps):
        if fr is not None and hasattr(fr, '_rms_norm_kernel'):
            try:
                shape = x.shape
                cols = shape[-1]
                x_f = x.view(-1, cols) if x.is_contiguous() else x.contiguous().view(-1, cols)
                out = torch.empty_like(x_f)
                import triton
                bs = triton.next_power_of_2(cols)
                fr._rms_norm_kernel[(x_f.shape[0],)](
                    x_f, weight, out, eps, cols,
                    x_f.stride(0), bs, num_warps=8 if bs >= 2048 else 4)
                return out.view(shape)
            except TypeError:
                pass
        return _rms_norm_eager(x, weight, eps)

    def _linear_fp8(x, w, s):
        xq, xs = _do_act_quant(x)
        return fp8_gemm(xq, xs, w, s)

    def _linear_fp8_reduce(x, w, s):
        y = _linear_fp8(x, w, s)
        y = y.float()
        dist.all_reduce(y)
        return y.to(x.dtype)

    max_t = model.max_seq_len
    finfo_min = torch.finfo(torch.float32).min
    static_attn_mask = torch.full((1, 1, 1, max_t), finfo_min, device="cuda", dtype=torch.float32)
    static_pos = torch.zeros(1, dtype=torch.long, device="cuda")
    static_fc = freqs_cis[0:1].clone()

    vocab_shard = head_weight.shape[0]
    static_logits = torch.empty(1, vocab_shard * ws, device="cuda", dtype=torch.float32)

    def _update_decode_buffers(pos: int):
        """Update static buffers before CUDA graph replay. Call from Python."""
        end_pos = pos + 1
        static_pos.fill_(pos)
        static_fc.copy_(freqs_cis[pos:pos+1])
        static_attn_mask.fill_(finfo_min)
        static_attn_mask[:, :, :, :end_pos] = 0.0

    @torch.inference_mode()
    def flat_decode_cg(tok):
        """Single decode step using static position buffers. CUDA-graph-safe."""
        bsz = 1
        fc = static_fc
        pos = static_pos[0]

        h = _cg_safe_embed(tok)
        residual = None

        for ld in layers:
            if residual is None:
                normed = _standalone_norm(h, ld['an_w'], ld['eps'])
                residual = h
            else:
                normed, residual = _residual_norm(h, residual, ld['an_w'], ld['eps'])

            if ld['wq_a_fp8']:
                xq, xs = _do_act_quant(normed)
                qr_raw = fp8_gemm(xq, xs, ld['wq_a_w'], ld['wq_a_s'])
                kv_raw = fp8_gemm(xq, xs, ld['wkv_a_w'], ld['wkv_a_s'])
            else:
                qr_raw = normed @ ld['wq_a_w'].T
                kv_raw = normed @ ld['wkv_a_w'].T

            qr = _standalone_norm(qr_raw, ld['q_norm_w'], ld['q_norm_eps'])
            q = _linear_fp8(qr, ld['wq_b_w'], ld['wq_b_s']) if ld['wq_b_fp8'] else qr @ ld['wq_b_w'].T
            q = q.view(bsz, 1, ld['n_heads'], ld['qk_head_dim'])
            q_nope, q_pe = q.split([ld['qk_nope_dim'], ld['qk_rope_dim']], dim=-1)
            q_pe = apply_rotary_emb(q_pe, fc)

            kv, k_pe = kv_raw.split([ld['kv_lora_rank'], ld['qk_rope_dim']], dim=-1)
            kv = _standalone_norm(kv, ld['kv_norm_w'], ld['kv_norm_eps'])
            k_pe = apply_rotary_emb(k_pe.unsqueeze(2), fc)

            pos_kv = static_pos.view(1, 1, 1).expand(bsz, 1, ld['kv_lora_rank'])
            ld['kv_cache'].scatter_(1, pos_kv, kv)
            pos_pe = static_pos.view(1, 1, 1).expand(bsz, 1, ld['qk_rope_dim'])
            ld['pe_cache'].scatter_(1, pos_pe, k_pe.squeeze(2))

            wkv_b = ld['wkv_b']
            q_nope_proj = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :ld['qk_nope_dim']])

            # Full cache read + mask (CUDA-graph-safe: fixed tensor sizes)
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope_proj, ld['kv_cache'][:bsz, :max_t])
                + torch.einsum("bshr,btr->bsht", q_pe, ld['pe_cache'][:bsz, :max_t])
            ) * ld['softmax_scale']
            scores = scores + static_attn_mask.to(scores.dtype)
            scores = scores.softmax(dim=-1)

            x = torch.einsum("bsht,btc->bshc", scores.to(ld['kv_cache'].dtype), ld['kv_cache'][:bsz, :max_t])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -ld['v_head_dim']:])
            attn_out = x.flatten(2)
            if ld['wo_fp8']:
                attn_out = _linear_fp8_reduce(attn_out, ld['wo_w'], ld['wo_s']) if ld['wo_reduce'] else _linear_fp8(attn_out, ld['wo_w'], ld['wo_s'])
            else:
                attn_out = ld['wo_fn'](attn_out)

            normed, residual = _residual_norm(attn_out, residual, ld['fn_w'], ld['eps'])

            if not ld['is_moe'] and ld.get('mlp_fp8'):
                gate_v = _linear_fp8(normed, ld['mlp_w1_w'], ld['mlp_w1_s'])
                up_v = _linear_fp8(normed, ld['mlp_w3_w'], ld['mlp_w3_s'])
                h_q, h_s = _do_swiglu_quant(gate_v, up_v)
                h = fp8_gemm(h_q, h_s, ld['mlp_w2_w'], ld['mlp_w2_s'])
                if ld['mlp_w2_reduce'] and ws > 1:
                    h = h.float()
                    dist.all_reduce(h)
                    h = h.to(torch.bfloat16)
            elif ld['is_moe'] and ld.get('moe_has_scales'):
                h = _flat_moe(normed, ld, _do_act_quant, _do_swiglu_quant, fp8_gemm, ws, _linear_fp8, _linear_fp8_reduce)
            else:
                h = ld['ffn_fn'](normed)

        if residual is not None:
            h_normed, _ = _residual_norm(h, residual, norm_weight, norm_eps)
        else:
            h_normed = _standalone_norm(h, norm_weight, norm_eps)

        s = model.head(h_normed[:, -1].float())
        if ws > 1:
            dist.all_gather_into_tensor(static_logits, s)
        else:
            static_logits.copy_(s)
        return static_logits

    @torch.inference_mode()
    def flat_decode(tok, start_pos):
        """Eager flat decode (non-CUDA-graph). Used for profiling."""
        _update_decode_buffers(start_pos)
        return flat_decode_cg(tok)

    return flat_decode, flat_decode_cg, _update_decode_buffers, static_logits


def _flat_moe(x, ld, do_act_quant, do_swiglu_quant, fp8_gemm, ws, linear_fp8, linear_fp8_reduce):
    """Inlined MoE with stacked weights. CUDA-graph-safe (no .tolist, no bincount)."""
    x_flat = x.view(-1, ld['moe_dim'])

    # Inline gate
    gw = ld['gate_weight']
    scores = torch.nn.functional.linear(x_flat.float(), gw.float())
    if ld['gate_score_func'] == "softmax":
        scores = scores.softmax(dim=-1)
    else:
        scores = scores.sigmoid()
    original_scores = scores
    if ld['gate_bias'] is not None:
        scores = scores + ld['gate_bias']
    if ld['gate_n_groups'] > 1:
        scores_g = scores.view(x_flat.size(0), ld['gate_n_groups'], -1)
        if ld['gate_bias'] is None:
            group_scores = scores_g.amax(dim=-1)
        else:
            group_scores = scores_g.topk(2, dim=-1)[0].sum(dim=-1)
        indices = group_scores.topk(ld['gate_topk_groups'], dim=-1)[1]
        mask = scores_g.new_ones(x_flat.size(0), ld['gate_n_groups'], dtype=torch.bool)
        mask.scatter_(1, indices, False)
        scores = scores_g.masked_fill_(mask.unsqueeze(-1), torch.finfo(scores.dtype).min).flatten(1)
    gate_indices = scores.topk(ld['gate_topk'], dim=-1)[1]
    weights = original_scores.gather(1, gate_indices)
    if ld['gate_score_func'] == "sigmoid":
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * ld['gate_route_scale']

    # Stacked expert dispatch
    start, n_local = ld['moe_start'], ld['moe_n_local']
    topk = ld['moe_topk']
    local_ids = (gate_indices[0] - start).clamp(0, n_local - 1).to(torch.long)
    is_local = ((gate_indices[0] >= start) & (gate_indices[0] < ld['moe_end'])).float()
    expert_weights = weights[0] * is_local

    sel_w1 = torch.index_select(ld['moe_w1'], 0, local_ids)
    sel_w3 = torch.index_select(ld['moe_w3'], 0, local_ids)
    sel_w2 = torch.index_select(ld['moe_w2'], 0, local_ids)
    sel_s1 = torch.index_select(ld['moe_w1_s'], 0, local_ids)
    sel_s3 = torch.index_select(ld['moe_w3_s'], 0, local_ids)
    sel_s2 = torch.index_select(ld['moe_w2_s'], 0, local_ids)

    y = torch.zeros_like(x_flat, dtype=torch.float32)
    for t in range(topk):
        x_q, x_s = do_act_quant(x_flat)
        gate_out = fp8_gemm(x_q, x_s, sel_w1[t], sel_s1[t])
        up_out = fp8_gemm(x_q, x_s, sel_w3[t], sel_s3[t])
        h_q, h_s = do_swiglu_quant(gate_out, up_out)
        out = fp8_gemm(h_q, h_s, sel_w2[t], sel_s2[t])
        y = y + out.float() * expert_weights[t]

    # Shared experts (inlined MLP)
    if ld.get('shared_fp8'):
        sg = linear_fp8(x_flat, ld['shared_mlp_w1_w'], ld['shared_mlp_w1_s'])
        su = linear_fp8(x_flat, ld['shared_mlp_w3_w'], ld['shared_mlp_w3_s'])
        sh_q, sh_s = do_swiglu_quant(sg, su)
        shared_out = fp8_gemm(sh_q, sh_s, ld['shared_mlp_w2_w'], ld['shared_mlp_w2_s'])
    else:
        shared_out = ld['shared_experts_fn'](x_flat)
    y = y + shared_out.float()

    if ws > 1:
        dist.all_reduce(y)
    return y.to(torch.bfloat16).view(x.shape)


def _get_or_alloc_cache(mla, name, dim):
    cache = getattr(mla, name, None)
    if cache is not None:
        return cache
    max_t = getattr(mla, 'max_seq_len', 4096)
    if hasattr(mla, 'kv_cache') and mla.kv_cache is not None:
        max_t = mla.kv_cache.shape[1]
    cache = torch.zeros(1, max_t, dim, device="cuda", dtype=torch.bfloat16)
    setattr(mla, name, cache)
    return cache


def _extract_layer(layer, aq, rn, ws):
    """Extract all parameters from a layer into a flat dict for fast access."""
    from inference.model import weight_dequant, MoE, MLP

    mla = layer.attn
    ffn = layer.ffn

    if mla.dequant_wkv_b is None and mla.wkv_b.scale is not None:
        mla.dequant_wkv_b = weight_dequant(mla.wkv_b.weight, mla.wkv_b.scale)
    wkv_b = mla.wkv_b.weight if mla.dequant_wkv_b is None else mla.dequant_wkv_b
    wkv_b = wkv_b.view(mla.n_local_heads, -1, mla.kv_lora_rank)

    ld = {
        'eps': layer.attn_norm.eps,
        'an_w': layer.attn_norm.weight,
        'fn_w': layer.ffn_norm.weight,
        'wq_a_w': mla.wq_a.weight, 'wq_a_s': mla.wq_a.scale,
        'wq_a_fp8': mla.wq_a.weight.dtype == torch.float8_e4m3fn,
        'wkv_a_w': mla.wkv_a.weight, 'wkv_a_s': mla.wkv_a.scale,
        'q_norm_w': mla.q_norm.weight, 'q_norm_eps': mla.q_norm.eps,
        'kv_norm_w': mla.kv_norm.weight, 'kv_norm_eps': mla.kv_norm.eps,
        'wq_b_w': mla.wq_b.weight, 'wq_b_s': mla.wq_b.scale,
        'wq_b_fp8': mla.wq_b.weight.dtype == torch.float8_e4m3fn,
        'wkv_b': wkv_b,
        'wo_w': mla.wo.weight, 'wo_s': mla.wo.scale,
        'wo_fp8': mla.wo.weight.dtype == torch.float8_e4m3fn,
        'wo_reduce': getattr(mla.wo, 'reduce_output', False),
        'wo_fn': mla.wo.forward,
        'kv_cache': _get_or_alloc_cache(mla, 'kv_cache', mla.kv_lora_rank),
        'pe_cache': _get_or_alloc_cache(mla, 'pe_cache', mla.qk_rope_head_dim),
        'n_heads': mla.n_local_heads,
        'qk_head_dim': mla.qk_head_dim,
        'qk_nope_dim': mla.qk_nope_head_dim,
        'qk_rope_dim': mla.qk_rope_head_dim,
        'kv_lora_rank': mla.kv_lora_rank,
        'v_head_dim': mla.v_head_dim,
        'softmax_scale': mla.softmax_scale,
        'index_topk': mla.indexer.index_topk,
        'indexer_fn': mla.indexer.forward,
        'is_moe': isinstance(ffn, MoE),
        'ffn_fn': ffn.forward,
    }

    if isinstance(ffn, MLP):
        ld['mlp_w1_w'] = ffn.w1.weight
        ld['mlp_w1_s'] = ffn.w1.scale
        ld['mlp_w3_w'] = ffn.w3.weight
        ld['mlp_w3_s'] = ffn.w3.scale
        ld['mlp_w2_w'] = ffn.w2.weight
        ld['mlp_w2_s'] = ffn.w2.scale
        ld['mlp_w2_reduce'] = getattr(ffn.w2, 'reduce_output', False)
        ld['mlp_fp8'] = ffn.w1.weight.dtype == torch.float8_e4m3fn

    if isinstance(ffn, MoE):
        gate = ffn.gate
        ld['gate_weight'] = gate.weight
        ld['gate_bias'] = gate.bias
        ld['gate_n_groups'] = gate.n_groups
        ld['gate_topk'] = gate.topk
        ld['gate_topk_groups'] = gate.topk_groups
        ld['gate_score_func'] = gate.score_func
        ld['gate_route_scale'] = gate.route_scale
        ld['moe_dim'] = ffn.dim
        ld['moe_topk'] = gate.topk
        start = ffn.experts_start_idx
        end = ffn.experts_end_idx
        ld['moe_start'] = start
        ld['moe_end'] = end
        local_experts = [ffn.experts[i] for i in range(start, end)]
        n_local = len(local_experts)
        ld['moe_n_local'] = n_local
        has_scales = (hasattr(local_experts[0].w1.weight, 'scale')
                      and local_experts[0].w1.weight.scale is not None)
        ld['moe_has_scales'] = has_scales
        def _incremental_stack(experts, attr_chain):
            """Stack weights one expert at a time to minimize peak memory.

            Pre-allocates the output, copies each expert's weight, then
            deletes the original before moving to the next. Peak memory
            is output + 1 expert weight instead of output + all experts.
            """
            parts = attr_chain.split(".")
            def _get(e):
                obj = e
                for p in parts:
                    obj = getattr(obj, p)
                return obj
            ref = _get(experts[0])
            out = torch.empty(len(experts), *ref.shape, dtype=ref.dtype, device=ref.device)
            for i, e in enumerate(experts):
                out[i].copy_(_get(e))
            return out

        def _clear_attr(experts, attr):
            for e in experts:
                if hasattr(e, attr):
                    setattr(e, attr, None)
            torch.cuda.empty_cache()

        if has_scales:
            ld['moe_w1_s'] = _incremental_stack(local_experts, "w1.weight.scale")
        ld['moe_w1'] = _incremental_stack(local_experts, "w1.weight.data")
        _clear_attr(local_experts, "w1")
        if has_scales:
            ld['moe_w3_s'] = _incremental_stack(local_experts, "w3.weight.scale")
        ld['moe_w3'] = _incremental_stack(local_experts, "w3.weight.data")
        _clear_attr(local_experts, "w3")
        if has_scales:
            ld['moe_w2_s'] = _incremental_stack(local_experts, "w2.weight.scale")
        ld['moe_w2'] = _incremental_stack(local_experts, "w2.weight.data")
        _clear_attr(local_experts, "w2")
        for i in range(start, end): ffn.experts[i] = None
        torch.cuda.empty_cache()
        ld['shared_experts_fn'] = ffn.shared_experts.forward
        ld['shared_mlp_w1_w'] = ffn.shared_experts.w1.weight
        ld['shared_mlp_w1_s'] = ffn.shared_experts.w1.scale
        ld['shared_mlp_w3_w'] = ffn.shared_experts.w3.weight
        ld['shared_mlp_w3_s'] = ffn.shared_experts.w3.scale
        ld['shared_mlp_w2_w'] = ffn.shared_experts.w2.weight
        ld['shared_mlp_w2_s'] = ffn.shared_experts.w2.scale
        ld['shared_mlp_w2_reduce'] = getattr(ffn.shared_experts.w2, 'reduce_output', False)
        ld['shared_fp8'] = ffn.shared_experts.w1.weight.dtype == torch.float8_e4m3fn

    return ld


@torch.inference_mode()
def generate_with_cudagraph(
    model,
    kernel_root: Path,
    prompt_tokens: list[list[int]],
    max_new_tokens: int,
    eos_id: int,
) -> list[list[int]]:
    """Full generation with CUDA graph for decode steps.

    Phase 1: Prefill using the original model (before MoE weight stacking).
    Phase 2: Build flat_decode (stacks MoE weights, destroys original experts).
    Phase 3: Warmup flat decode, then capture CUDA graph.
    Phase 4: Generate tokens using CUDA graph replay.

    WARNING: This destructively modifies the model (stacks MoE weights,
    sets original experts to None). The model cannot be used for normal
    inference after this function returns.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    max_seq_len = model.max_seq_len
    total_len = min(max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full(
        (len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda",
    )
    for i, t in enumerate(prompt_tokens):
        tokens[i, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    min_prompt = min(prompt_lens)

    # Phase 1: Prefill with original model (experts still intact)
    model.forward(tokens[:, :min_prompt], 0)

    # Phase 2: Build flat decode (stacks MoE weights, destroys experts)
    flat_fn, flat_cg_fn, update_bufs, static_logits = build_flat_decode(model, kernel_root)

    # Phase 3: Warmup + capture CUDA graph
    static_tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    warmup_pos = min_prompt
    for i in range(3):
        update_bufs(warmup_pos + i)
        static_tok.fill_(tokens[0, warmup_pos + i].item() if warmup_pos + i < total_len else 0)
        flat_cg_fn(static_tok)
    torch.cuda.synchronize()

    graph = None
    try:
        update_bufs(warmup_pos + 3)
        static_tok.fill_(0)
        flat_cg_fn(static_tok)
        torch.cuda.synchronize()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            flat_cg_fn(static_tok)
        torch.cuda.synchronize()
        graph = cg
    except Exception:
        pass

    # Phase 4: Generate tokens
    prev_pos = min_prompt - 1
    for cur_pos in range(min_prompt, total_len):
        if graph is not None and (cur_pos - prev_pos) == 1:
            static_tok.fill_(tokens[0, prev_pos].item())
            update_bufs(prev_pos)
            graph.replay()
            logits = static_logits
        else:
            update_bufs(prev_pos)
            logits = flat_fn(tokens[:, prev_pos:cur_pos], prev_pos)

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

    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i] : prompt_lens[i] + max_new_tokens]
        if eos_id in toks:
            toks = toks[: toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens
