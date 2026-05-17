"""CUDA graph decode with fused Triton kernels + batched MoE.

Combines: kernel fusion (54k→fewer launches) + CUDA graph (zero dispatch overhead).

Run with:
  PYTHONPATH=~/local/b/pytorch:. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --standalone --nproc-per-node=8 /tmp/test_cg_fused.py
"""
import os, sys, time, importlib.util, torch, torch.nn.functional as F, torch.distributed as dist

# Load fused kernels directly (no dispatcher)
def _load_kernel(name):
    path = f"partitions/dsv3_2/3336cdbd/kernels/triton/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

KERNELS = {}

def load_kernels():
    KERNELS['residual_norm'] = _load_kernel('residual_norm')
    KERNELS['swiglu'] = _load_kernel('swiglu')
    KERNELS['act_quant'] = _load_kernel('act_quant')
    KERNELS['rope'] = _load_kernel('rope')
    KERNELS['fused_rope'] = _load_kernel('fused_rope')  # has standalone _rms_norm_kernel
    KERNELS['norm_act_quant'] = _load_kernel('norm_act_quant')
    KERNELS['rope_inline'] = _load_kernel('rope_inline')


def main():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)

    from benchmarks.gsm8k.hf_model_loader import load_hf_sharded_weights, run_post_load_transforms
    from benchmarks.gsm8k.eval import DSV3_2_CONFIG
    from inference.model import ModelArgs, Transformer
    from benchmarks.gsm8k.hf_auth import require_hf_token

    config = dict(DSV3_2_CONFIG)
    config["max_batch_size"] = 1
    config["max_seq_len"] = 256
    config["dtype"] = "fp8"
    args = ModelArgs(**config)
    hf_token = require_hf_token(None, purpose="cg fused")

    if rank == 0: print("Loading model...", flush=True)
    with torch.device("cuda"):
        model = Transformer(args)
    load_hf_sharded_weights(model, repo_id="deepseek-ai/DeepSeek-V3.2",
                            hf_token=hf_token, rank=rank, world_size=world_size)
    run_post_load_transforms(model)
    model.eval()
    load_kernels()

    if rank == 0: print("Patching with fused kernels...", flush=True)
    patch_for_cudagraph(model)
    torch.cuda.synchronize()
    if rank == 0:
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"Memory: {mem:.1f} GB", flush=True)

    with torch.inference_mode():
        tok = torch.arange(16, device="cuda", dtype=torch.long).unsqueeze(0)
        model.forward(tok, 0)
        for i in range(10):
            model.forward(torch.tensor([[100+i]], device="cuda", dtype=torch.long), 16+i)
        torch.cuda.synchronize()

        N = 50
        start = 26

        # Eager baseline (unpatched model won't work since we patched it)
        # Just time the patched model eagerly
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(N):
            model.forward(torch.tensor([[200+i]], device="cuda", dtype=torch.long), start+i)
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) * 1000 / N
        if rank == 0: print(f"Fused eager decode: {eager_ms:.2f} ms/token", flush=True)

        # CUDA graph capture
        static_tok = torch.tensor([[0]], device="cuda", dtype=torch.long)
        vocab_shard = model.head.weight.shape[0]
        static_logits = torch.empty(1, vocab_shard * world_size, device="cuda", dtype=torch.float32)
        capture_pos = start + N // 2

        def decode_step():
            fc = model.freqs_cis[capture_pos:capture_pos+1]
            h = model.embed(static_tok)
            residual = None
            for layer in model.layers:
                h, residual = layer(h, residual, capture_pos, fc, None)
            h, _ = model.norm(h, residual)
            s = model.head(h[:, -1].float())
            dist.all_gather_into_tensor(static_logits, s)
            return static_logits

        decode_step(); torch.cuda.synchronize()
        decode_step(); torch.cuda.synchronize()

        if rank == 0: print("Capturing CUDA graph...", flush=True)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                result = decode_step()
            torch.cuda.synchronize()
            if rank == 0: print("Captured!", flush=True)
        except Exception as e:
            if rank == 0: print(f"Capture failed: {e}", flush=True)
            if dist.is_available() and dist.is_initialized():
                dist.barrier(); dist.destroy_process_group()
            return

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(N):
            static_tok.fill_(300+i)
            graph.replay()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - t0) * 1000 / N

        if rank == 0:
            print(f"CUDA graph + fused kernels: {graph_ms:.2f} ms/token", flush=True)

        # Re-run timing to confirm
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(N):
            static_tok.fill_(500+i)
            graph.replay()
        torch.cuda.synchronize()
        graph_ms2 = (time.perf_counter() - t0) * 1000 / N
        if rank == 0:
            print(f"CUDA graph (recheck): {graph_ms2:.2f} ms/token", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()


def patch_for_cudagraph(model):
    from inference import model as rm
    for module in model.modules():
        if isinstance(module, rm.Indexer):
            _patch_indexer_shortcircuit(module)
        if isinstance(module, rm.MoE):
            _patch_moe(module)
        if isinstance(module, rm.Gate):
            _patch_gate(module)
        if isinstance(module, rm.MLA):
            _patch_mla(module)
        if isinstance(module, rm.RMSNorm):
            _patch_rmsnorm(module)
        if isinstance(module, rm.MLP):
            _patch_mlp(module)
        if isinstance(module, (rm.Linear, rm.ColumnParallelLinear, rm.RowParallelLinear)):
            if module.weight.dtype == torch.float8_e4m3fn:
                _patch_linear_global(module)


def _patch_indexer_shortcircuit(module):
    """Skip entire indexer when seq < topk — return arange(end_pos)."""
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

    module.forward = forward


def _patch_linear_global(module):
    """Patch ALL FP8 Linear modules: fused_act_quant + fp8_gemm, no redundant casts."""
    aq = KERNELS['act_quant']
    def forward(x):
        from inference.kernel import fp8_gemm
        xq, xs = aq.fused_act_quant(x, fallback=None)
        y = fp8_gemm(xq, xs, module.weight, module.weight.scale)
        if getattr(module, 'reduce_output', False) and dist.is_initialized() and dist.get_world_size() > 1:
            y = y.float(); dist.all_reduce(y)
            return y.to(x.dtype)
        return y
    module.forward = forward


def _fused_standalone_norm(x, weight, eps):
    """Fused standalone RMSNorm — 1 Triton kernel instead of 5 eager ops."""
    rope_mod = KERNELS.get('fused_rope')
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


def _patch_rmsnorm(module):
    """Use fused Triton kernels for both residual and standalone norm."""
    kernel = KERNELS['residual_norm']
    original = module.forward

    def forward(x, residual=None):
        if residual is None:
            return _fused_standalone_norm(x, module.weight, module.eps)
        # Call fused kernel directly — no dispatcher
        shape = x.shape
        cols = shape[-1]
        x_flat = x.view(-1, cols) if x.is_contiguous() else x.contiguous().view(-1, cols)
        r_flat = residual.view(-1, cols) if residual.is_contiguous() else residual.contiguous().view(-1, cols)
        hidden, normed = kernel.fused_residual_norm(
            r_flat, x_flat, module.weight, eps=module.eps, fallback=None,
        )
        return normed.view(shape), hidden.view(shape)

    module.forward = forward


def _patch_mlp(module):
    """Use fused kernels for all Linear calls + swiglu+quant."""
    aq = KERNELS['act_quant']
    w2_is_fp8 = module.w2.weight.dtype == torch.float8_e4m3fn

    def forward(x):
        from inference.kernel import fp8_gemm
        gate = _fused_linear(x, module.w1, aq)
        up = _fused_linear(x, module.w3, aq)
        if w2_is_fp8:
            # swiglu + act_quant in one kernel → skip separate type_cast + act_quant
            h_fp8, h_s = aq.fused_swiglu_quant(gate, up, fallback=None)
            y = fp8_gemm(h_fp8, h_s, module.w2.weight, module.w2.weight.scale)
            if getattr(module.w2, 'reduce_output', False) and dist.is_initialized() and dist.get_world_size() > 1:
                y = y.float(); dist.all_reduce(y)
                return y.to(x.dtype)
            return y  # bf16 already
        swiglu_kernel = KERNELS['swiglu']
        hidden = swiglu_kernel.fused_swiglu(gate, up, fallback=None)
        return _fused_linear(hidden, module.w2, aq)  # no .type_as() needed

    module.forward = forward


def _patch_moe(module):
    """Batched MoE with stacked weights + fused kernels."""
    topk = module.gate.topk
    start = module.experts_start_idx
    end = module.experts_end_idx
    local_experts = [module.experts[i] for i in range(start, end)]
    n_local = len(local_experts)

    has_scales = hasattr(local_experts[0].w1.weight, 'scale') and local_experts[0].w1.weight.scale is not None
    scale_fmt = local_experts[0].w1.scale_fmt if has_scales else None

    if has_scales:
        w1_s = torch.stack([e.w1.weight.scale for e in local_experts])
    w1_stack = torch.stack([e.w1.weight.data for e in local_experts])
    for e in local_experts: e.w1 = None
    torch.cuda.empty_cache()

    if has_scales:
        w3_s = torch.stack([e.w3.weight.scale for e in local_experts])
    w3_stack = torch.stack([e.w3.weight.data for e in local_experts])
    for e in local_experts: e.w3 = None
    torch.cuda.empty_cache()

    if has_scales:
        w2_s = torch.stack([e.w2.weight.scale for e in local_experts])
    w2_stack = torch.stack([e.w2.weight.data for e in local_experts])
    for e in local_experts: e.w2 = None
    torch.cuda.empty_cache()

    for i in range(start, end):
        module.experts[i] = None
    torch.cuda.empty_cache()

    swiglu_kernel = KERNELS['swiglu']
    aq_kernel = KERNELS['act_quant']

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

            tc = KERNELS.get('fp8_gemm_tc')
            for t in range(topk):
                if tc is not None:
                    gate_out = tc.fp8_gemm_tc(x_flat, sel_w1[t], sel_s1[t])
                    up_out = tc.fp8_gemm_tc(x_flat, sel_w3[t], sel_s3[t])
                    hidden = swiglu_kernel.fused_swiglu(gate_out, up_out, fallback=None)
                    out = tc.fp8_gemm_tc(hidden, sel_w2[t], sel_s2[t])
                else:
                    x_q, x_s = aq_kernel.fused_act_quant(x_flat, fallback=None)
                    gate_out = fp8_gemm(x_q, x_s, sel_w1[t], sel_s1[t])
                    up_out = fp8_gemm(x_q, x_s, sel_w3[t], sel_s3[t])
                    h_q, h_s = aq_kernel.fused_swiglu_quant(gate_out, up_out, fallback=None)
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

    module.forward = forward


def _patch_gate(module):
    def forward(x):
        from inference.model import linear
        scores = linear(x.float(), module.weight.float())
        if module.score_func == "softmax": scores = scores.softmax(dim=-1)
        else: scores = scores.sigmoid()
        original_scores = scores
        if module.bias is not None: scores = scores + module.bias
        if module.n_groups > 1:
            scores = scores.view(x.size(0), module.n_groups, -1)
            if module.bias is None: group_scores = scores.amax(dim=-1)
            else: group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(module.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), module.n_groups, dtype=torch.bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), torch.finfo(scores.dtype).min).flatten(1)
        indices = scores.topk(module.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if module.score_func == "sigmoid": weights /= weights.sum(dim=-1, keepdim=True)
        weights *= module.route_scale
        return weights, indices
    module.forward = forward


def _fused_linear(x, linear_mod, aq_kernel):
    """Replace Linear.forward: fused_act_quant + fp8_gemm, minimal casts."""
    if linear_mod.weight.dtype == torch.float8_e4m3fn:
        from inference.kernel import fp8_gemm
        x_q, x_s = aq_kernel.fused_act_quant(x, fallback=None)
        y = fp8_gemm(x_q, x_s, linear_mod.weight, linear_mod.weight.scale)
        if getattr(linear_mod, 'reduce_output', False) and dist.is_initialized() and dist.get_world_size() > 1:
            y = y.float()
            dist.all_reduce(y)
            return y.to(x.dtype)
        return y  # fp8_gemm outputs bf16, same as x — no cast needed
    return linear_mod(x)


def _patch_mla(module):
    original = module.forward
    aq = KERNELS['act_quant']

    def forward(x, start_pos, freqs_cis, mask):
        if mask is not None: return original(x, start_pos, freqs_cis, mask)
        from inference.model import apply_rotary_emb as _orig_rope, weight_dequant
        ri = KERNELS.get('rope_inline')
        def apply_rope(t, fc, interleaved=True):
            if ri and interleaved:
                return ri.rope_inline(t, fc)
            return _orig_rope(t, fc, interleaved)
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # Fused act_quant for wq_a — shared with wkv_a (same input x)
        from inference.kernel import fp8_gemm

        # Share act_quant(x) for BOTH wq_a and wkv_a (same input)
        # Inductor does this in 1 kernel; we do 1 quant + 2 GEMMs
        if module.wq_a.weight.dtype == torch.float8_e4m3fn:
            x_q, x_s = aq.fused_act_quant(x, fallback=None)
            qr_raw = fp8_gemm(x_q, x_s, module.wq_a.weight, module.wq_a.weight.scale)
            kv_raw = fp8_gemm(x_q, x_s, module.wkv_a.weight, module.wkv_a.weight.scale)
        else:
            qr_raw = module.wq_a(x)
            kv_raw = module.wkv_a(x)

        # Q path: norm → quant → GEMM
        naq = KERNELS.get('norm_act_quant')
        if naq and module.wq_b.weight.dtype == torch.float8_e4m3fn:
            qr_fp8, qr_scale = naq.fused_standalone_norm_quant(
                qr_raw, module.q_norm.weight, eps=module.q_norm.eps,
            )
            q = fp8_gemm(qr_fp8, qr_scale, module.wq_b.weight, module.wq_b.weight.scale)
        else:
            qr = _fused_standalone_norm(qr_raw, module.q_norm.weight, module.q_norm.eps)
            q = _fused_linear(qr, module.wq_b, aq)
        q = q.view(bsz, seqlen, module.n_local_heads, module.qk_head_dim)
        q_nope, q_pe = torch.split(q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1)
        q_pe = apply_rope(q_pe, freqs_cis)

        # KV path: reuse kv_raw from shared quant above
        kv = kv_raw
        kv, k_pe = torch.split(kv, [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1)
        kv = _fused_standalone_norm(kv, module.kv_norm.weight, module.kv_norm.eps)
        k_pe = apply_rope(k_pe.unsqueeze(2), freqs_cis)
        module.kv_cache[:bsz, start_pos:end_pos] = kv
        module.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

        if module.dequant_wkv_b is None and module.wkv_b.scale is not None:
            module.dequant_wkv_b = weight_dequant(module.wkv_b.weight, module.wkv_b.scale)
        wkv_b = module.wkv_b.weight if module.dequant_wkv_b is None else module.dequant_wkv_b
        wkv_b = wkv_b.view(module.n_local_heads, -1, module.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :module.qk_nope_head_dim])
        scores = (torch.einsum("bshc,btc->bsht", q_nope, module.kv_cache[:bsz, :end_pos]) +
                  torch.einsum("bshr,btr->bsht", q_pe, module.pe_cache[:bsz, :end_pos])) * module.softmax_scale
        if end_pos <= module.indexer.index_topk:
            # Full-topk shortcircuit: ALL positions selected, no masking needed
            scores = scores.softmax(dim=-1)
        else:
            topk_indices = module.indexer(x, qr_raw, start_pos, freqs_cis, None)
            index_mask = scores.new_full((bsz, 1, end_pos), torch.finfo(scores.dtype).min).scatter_(-1, topk_indices, 0.0)
            scores += index_mask.unsqueeze(2)
            scores = scores.softmax(dim=-1)
        x = torch.einsum("bsht,btc->bshc", scores, module.kv_cache[:bsz, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -module.v_head_dim:])
        return _fused_linear(x.flatten(2), module.wo, aq)
    module.forward = forward


if __name__ == "__main__":
    main()
