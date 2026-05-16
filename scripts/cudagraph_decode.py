"""CUDA graph decode with batched MoE experts.

Run with:
  PYTHONPATH=~/local/b/pytorch:. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --standalone --nproc-per-node=8 /tmp/test_cg_full.py
"""
import os, time, torch, torch.nn.functional as F, torch.distributed as dist

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
    hf_token = require_hf_token(None, purpose="cg full")

    if rank == 0: print("Loading model...", flush=True)
    with torch.device("cuda"):
        model = Transformer(args)
    load_hf_sharded_weights(model, repo_id="deepseek-ai/DeepSeek-V3.2",
                            hf_token=hf_token, rank=rank, world_size=world_size)
    run_post_load_transforms(model)
    model.eval()

    # Patch BEFORE any forward (before KV caches allocate)
    if rank == 0: print("Patching for CUDA graph...", flush=True)
    patch_for_cudagraph(model)
    torch.cuda.synchronize()
    if rank == 0:
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"Memory after patching: {mem:.1f} GB", flush=True)

    with torch.inference_mode():
        # Prefill + warmup
        tok = torch.arange(16, device="cuda", dtype=torch.long).unsqueeze(0)
        model.forward(tok, 0)
        for i in range(10):
            model.forward(torch.tensor([[100+i]], device="cuda", dtype=torch.long), 16+i)
        torch.cuda.synchronize()

        # Eager baseline
        N = 50
        start = 26
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(N):
            model.forward(torch.tensor([[200+i]], device="cuda", dtype=torch.long), start+i)
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) * 1000 / N
        if rank == 0: print(f"Eager decode (patched): {eager_ms:.2f} ms/token", flush=True)

        # Capture
        static_tok = torch.tensor([[0]], device="cuda", dtype=torch.long)
        vocab_shard = model.head.weight.shape[0]
        static_logits = torch.empty(1, vocab_shard * world_size, device="cuda", dtype=torch.float32)
        capture_pos = start + N

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

        # Benchmark replay
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(N):
            static_tok.fill_(300+i)
            graph.replay()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - t0) * 1000 / N

        if rank == 0:
            print(f"CUDA graph replay: {graph_ms:.2f} ms/token", flush=True)
            print(f"Speedup vs eager: {eager_ms/graph_ms:.3f}x", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()


def patch_for_cudagraph(model):
    from inference import model as rm
    for module in model.modules():
        if isinstance(module, rm.MoE):
            _patch_moe(module)
        if isinstance(module, rm.Gate):
            _patch_gate(module)
        if isinstance(module, rm.MLA):
            _patch_mla(module)


def _patch_moe(module):
    """
    Batch experts into single weight tensors for CUDA-tensor indexing.
    Runs only topk experts per token without CPU sync.
    """
    topk = module.gate.topk
    start = module.experts_start_idx
    end = module.experts_end_idx
    local_experts = [module.experts[i] for i in range(start, end)]
    n_local = len(local_experts)

    # Stack expert weights one matrix at a time, freeing originals after each
    # to keep peak overhead to ~470MB instead of ~1.4GB
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

    # Clear expert module references
    for i in range(start, end):
        module.experts[i] = None
    torch.cuda.empty_cache()

    def forward(x):
        from inference.kernel import act_quant, fp8_gemm
        from inference.model import block_size

        shape = x.size()
        x_flat = x.view(-1, module.dim)
        weights, indices = module.gate(x_flat)

        y = torch.zeros_like(x_flat, dtype=torch.float32)

        for t in range(topk):
            eid = indices[0, t]
            local_id = (eid - start).clamp(0, n_local - 1)
            is_local = ((eid >= start) & (eid < end)).float()
            w = weights[0:1, t:t+1] * is_local

            idx = local_id.view(1).to(torch.long)
            ew1 = torch.index_select(w1_stack, 0, idx).squeeze(0)
            ew3 = torch.index_select(w3_stack, 0, idx).squeeze(0)
            ew2 = torch.index_select(w2_stack, 0, idx).squeeze(0)

            if has_scales:
                es1 = torch.index_select(w1_s, 0, idx).squeeze(0)
                es3 = torch.index_select(w3_s, 0, idx).squeeze(0)
                es2 = torch.index_select(w2_s, 0, idx).squeeze(0)
                x_q, x_s = act_quant(x_flat, block_size, scale_fmt)
                gate_out = fp8_gemm(x_q, x_s, ew1, es1)
                up_out = fp8_gemm(x_q, x_s, ew3, es3)
                hidden = (F.silu(gate_out.float()) * up_out.float()).to(gate_out.dtype)
                h_q, h_s = act_quant(hidden, block_size, scale_fmt)
                out = fp8_gemm(h_q, h_s, ew2, es2)
            else:
                gate_out = F.linear(x_flat, ew1)
                up_out = F.linear(x_flat, ew3)
                hidden = (F.silu(gate_out.float()) * up_out.float()).to(gate_out.dtype)
                out = F.linear(hidden, ew2)

            y += out.float() * w

        y += module.shared_experts(x_flat)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(y)
        return y.type_as(x).view(shape)

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


def _patch_mla(module):
    original = module.forward
    def forward(x, start_pos, freqs_cis, mask):
        if mask is not None: return original(x, start_pos, freqs_cis, mask)
        from inference.model import apply_rotary_emb, weight_dequant
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        qr = module.q_norm(module.wq_a(x))
        q = module.wq_b(qr)
        q = q.view(bsz, seqlen, module.n_local_heads, module.qk_head_dim)
        q_nope, q_pe = torch.split(q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = module.wkv_a(x)
        kv, k_pe = torch.split(kv, [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1)
        kv = module.kv_norm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        module.kv_cache[:bsz, start_pos:end_pos] = kv
        module.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        if module.dequant_wkv_b is None and module.wkv_b.scale is not None:
            module.dequant_wkv_b = weight_dequant(module.wkv_b.weight, module.wkv_b.scale)
        wkv_b = module.wkv_b.weight if module.dequant_wkv_b is None else module.dequant_wkv_b
        wkv_b = wkv_b.view(module.n_local_heads, -1, module.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :module.qk_nope_head_dim])
        scores = (torch.einsum("bshc,btc->bsht", q_nope, module.kv_cache[:bsz, :end_pos]) +
                  torch.einsum("bshr,btr->bsht", q_pe, module.pe_cache[:bsz, :end_pos])) * module.softmax_scale
        topk_indices = module.indexer(x, qr, start_pos, freqs_cis, None)
        index_mask = scores.new_full((bsz, 1, end_pos), torch.finfo(scores.dtype).min).scatter_(-1, topk_indices, 0.0)
        scores += index_mask.unsqueeze(2)
        scores = scores.softmax(dim=-1)
        x = torch.einsum("bsht,btc->bshc", scores, module.kv_cache[:bsz, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -module.v_head_dim:])
        return module.wo(x.flatten(2))
    module.forward = forward


if __name__ == "__main__":
    main()
