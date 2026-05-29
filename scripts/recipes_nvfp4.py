"""Fusion recipes for the dsv3_2_nvfp4 model.

The 4 NVFP4 fusions are deeply interdependent — the fused_ar_rms_qkv_proj
recipe restructures the block/attention interface, and the other 3 recipes
add dispatch calls inside that restructured attention forward. All 4 must
be specified together.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NVFP4_BASELINE = PROJECT_ROOT / "partitions" / "dsv3_2_nvfp4" / "model.py"
NVFP4_PARTITIONS_DIR = PROJECT_ROOT / "partitions" / "dsv3_2_nvfp4"

NVFP4_NODE_IDS = {
    "input_ids", "embed", "ar_add_rms", "qkv_a_proj", "indexer_k_proj",
    "q_rms", "q_b_proj", "q_rope", "cat_q", "q_quant_fp8",
    "kv_c_rms", "kv_rope", "kv_quant_fp8", "mla_cache",
    "indexer_ln", "indexer_rope", "indexer_quant_fp8", "indexer_cache",
    "indexer_w", "indexer_q_proj", "indexer_q_rope", "indexer_q_fp8",
    "w_uk_t", "indexer_w_scale", "indexer_mqa",
    "logits_topk", "topk_page_idx", "mla", "w_uv", "o_proj",
    "ffn_norm", "gate_router", "topk_softmax",
    "w1_proj", "w3_proj", "swiglu", "w2_proj", "expert_sum",
    "res_add_ffn", "final_norm", "lm_head", "logits",
}

NVFP4_RUNTIME_OPS: set[str] = set()


# ---------------------------------------------------------------------------
# Extra ops (fused fallback functions)
# ---------------------------------------------------------------------------

_AR_RMS_QKV_OPS = """\
def fused_ar_rms_qkv_proj(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    norm_weight: torch.Tensor,
    wq_a_weight: torch.Tensor,
    wkv_a_weight: torch.Tensor,
    indexer_wk_weight: torch.Tensor,
    *,
    eps: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if residual is not None:
        hidden = x.float() + residual.float()
    else:
        hidden = x.float()
    dtype = x.dtype
    var = hidden.square().mean(dim=-1, keepdim=True)
    normed = (norm_weight.float() * hidden * torch.rsqrt(var + eps)).to(dtype)
    residual_out = hidden.to(dtype)
    qkv = F.linear(normed, torch.cat([wq_a_weight, wkv_a_weight], dim=0))
    total = q_lora_rank + kv_lora_rank + qk_rope_head_dim
    q_c = qkv[..., :q_lora_rank]
    kv_c = qkv[..., q_lora_rank:q_lora_rank + kv_lora_rank]
    k_pe = qkv[..., q_lora_rank + kv_lora_rank:total]
    indexer_k = F.linear(normed, indexer_wk_weight)
    return residual_out, normed, q_c, kv_c, k_pe, indexer_k"""

_INDEXER_K_OPS = """\
def fused_indexer_k_path(
    k: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    freqs_cis: torch.Tensor,
    H: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    *,
    ln_dim: int,
    ln_eps: float,
    rope_head_dim: int,
    start_pos: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = layer_norm(k, ln_weight, ln_bias, ln_dim, ln_eps)
    k_pe, k_nope = torch.split(k, [rope_head_dim, k.shape[-1] - rope_head_dim], dim=-1)
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)
    k = hadamard_transform(k, H)
    k_fp8, k_scale = act_quant(k, block_size)
    bsz, seqlen = k_fp8.shape[0], k_fp8.shape[1]
    end_pos = start_pos + seqlen
    k_cache[:bsz, start_pos:end_pos] = k_fp8.float()
    k_scale_cache[:bsz, start_pos:end_pos] = k_scale
    return k_fp8, k_scale"""

_Q_INDEXER_SCORE_OPS = """\
def fused_q_indexer_score(
    qr: torch.Tensor,
    normed_x: torch.Tensor,
    wq_b_weight: torch.Tensor,
    idx_wq_b_weight: torch.Tensor,
    idx_weights_proj_weight: torch.Tensor,
    wkv_b_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    H: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    *,
    eps: float,
    n_heads: int,
    qk_head_dim: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    idx_n_heads: int,
    idx_head_dim: int,
    rope_head_dim: int,
    softmax_scale: float,
    idx_softmax_scale: float,
    index_topk: int,
    start_pos: int,
    end_pos: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seqlen, _ = qr.shape
    q = F.linear(qr, wq_b_weight).view(bsz, seqlen, n_heads, qk_head_dim)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_head_dim - qk_nope_head_dim], dim=-1)
    wkv_b = wkv_b_weight.view(n_heads, -1, kv_lora_rank)
    q_nope_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope_head_dim])
    idx_q = F.linear(qr, idx_wq_b_weight).view(bsz, seqlen, idx_n_heads, idx_head_dim)
    idx_q_pe, idx_q_nope = torch.split(idx_q, [rope_head_dim, idx_head_dim - rope_head_dim], dim=-1)
    idx_q_pe = apply_rotary_emb(idx_q_pe, freqs_cis, interleaved=False)
    idx_q = torch.cat([idx_q_pe, idx_q_nope], dim=-1)
    idx_q = hadamard_transform(idx_q, H)
    idx_q_fp8, idx_q_scale = act_quant(idx_q, block_size)
    weights = F.linear(normed_x.float(), idx_weights_proj_weight.float()) * idx_n_heads ** -0.5
    weights = (weights.unsqueeze(-1) * idx_q_scale * idx_softmax_scale).squeeze(-1)
    k_s = k_scale_cache[:bsz, :end_pos].squeeze(-1).contiguous()
    k_cached = k_cache[:bsz, :end_pos].contiguous()
    index_score = fp8_index(idx_q_fp8.float(), weights, k_cached, k_s)
    topk_indices = index_score.topk(min(index_topk, end_pos), dim=-1)[1]
    return q_nope, q_nope_absorbed, q_pe, topk_indices"""

_Q_ROPE_QUANT_OPS = """\
def fused_q_rope_quant(
    q_pe: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=True)
    return q_pe"""


# ---------------------------------------------------------------------------
# Model patches
# ---------------------------------------------------------------------------

_BLOCK_FORWARD_OLD = """\
        if residual is None:
            x, residual = self.attn_norm(x), x
        else:
            x, residual = self.attn_norm(x, residual)
        x = self.attn(x, start_pos, freqs_cis, mask)
        x, residual = self.ffn_norm(x, residual)
        x = self.ffn(x)
        return x, residual"""

_BLOCK_FORWARD_NEW = """\
        config = self.config

        if self.dispatcher is None:
            residual_out, normed, q_c, kv_c, k_pe, indexer_k = fused_ar_rms_qkv_proj(
                x, residual, self.attn_norm.weight,
                self.attn.wq_a.weight, self.attn.wkv_a.weight,
                self.attn.indexer.wk.weight,
                eps=config.rms_norm_eps,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_rope_head_dim=config.qk_rope_head_dim,
            )
        else:
            residual_out, normed, q_c, kv_c, k_pe, indexer_k = self.dispatcher.call(
                "fused_ar_rms_qkv_proj", fused_ar_rms_qkv_proj,
                x, residual, self.attn_norm.weight,
                self.attn.wq_a.weight, self.attn.wkv_a.weight,
                self.attn.indexer.wk.weight,
                eps=config.rms_norm_eps,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_rope_head_dim=config.qk_rope_head_dim,
            )

        x = self.attn(normed, q_c, kv_c, k_pe, indexer_k, start_pos, freqs_cis, mask)
        x, residual = self.ffn_norm(x, residual_out)
        x = self.ffn(x)
        return x, residual"""

_ATTN_FORWARD_OLD = """\
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=True)

        kv = self.wkv_a(x)
        kv_c, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c = self.kv_norm(kv_c)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True)

        self.kv_cache[:bsz, start_pos:end_pos] = kv_c
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

        topk_indices = self.indexer(x, qr, start_pos, freqs_cis)

        if mask is not None:
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv_expanded = self.wkv_b(kv_c)
            kv_expanded = kv_expanded.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

            scores = torch.einsum("bshd,bthd->bsht", q.float(), k.float()) * self.softmax_scale
            index_mask = torch.full(
                (bsz, seqlen, seqlen), float("-inf"), device=x.device,
            ).scatter_(-1, topk_indices, 0)
            scores = scores + (index_mask + mask).unsqueeze(2)
            scores = scores.softmax(dim=-1).to(v.dtype)
            x = torch.einsum("bsht,bthd->bshd", scores, v)
        else:
            wkv_b = self.wkv_b.weight.view(self.n_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
                + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            ) * self.softmax_scale

            index_mask = torch.full(
                (bsz, 1, end_pos), float("-inf"), device=x.device,
            ).scatter_(-1, topk_indices, 0)
            scores = scores + index_mask.unsqueeze(2)
            scores = scores.softmax(dim=-1)

            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])

        x = self.wo(x.flatten(2))
        return x"""

_ATTN_FORWARD_NEW = """\
    def forward(
        self,
        normed_x: torch.Tensor,
        q_c: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        indexer_k: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = normed_x.size()
        end_pos = start_pos + seqlen
        config = self.config

        qr = self.q_norm(q_c)

        if self.dispatcher is None:
            fused_indexer_k_path(
                indexer_k,
                self.indexer.k_norm.weight, self.indexer.k_norm.bias,
                freqs_cis, self.indexer.hadamard_matrix,
                self.indexer.k_cache, self.indexer.k_scale_cache,
                ln_dim=self.indexer.head_dim, ln_eps=self.indexer.k_norm.eps,
                rope_head_dim=config.qk_rope_head_dim,
                start_pos=start_pos, block_size=config.block_size,
            )
        else:
            self.dispatcher.call(
                "fused_indexer_k_path", fused_indexer_k_path,
                indexer_k,
                self.indexer.k_norm.weight, self.indexer.k_norm.bias,
                freqs_cis, self.indexer.hadamard_matrix,
                self.indexer.k_cache, self.indexer.k_scale_cache,
                ln_dim=self.indexer.head_dim, ln_eps=self.indexer.k_norm.eps,
                rope_head_dim=config.qk_rope_head_dim,
                start_pos=start_pos, block_size=config.block_size,
            )

        if self.dispatcher is None:
            q_nope, q_nope_absorbed, q_pe, topk_indices = fused_q_indexer_score(
                qr, normed_x,
                self.wq_b.weight, self.indexer.wq_b.weight,
                self.indexer.weights_proj.weight,
                self.wkv_b.weight,
                freqs_cis, self.indexer.hadamard_matrix,
                self.indexer.k_cache, self.indexer.k_scale_cache,
                eps=config.rms_norm_eps,
                n_heads=self.n_heads, qk_head_dim=self.qk_head_dim,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                idx_n_heads=self.indexer.n_heads,
                idx_head_dim=self.indexer.head_dim,
                rope_head_dim=config.qk_rope_head_dim,
                softmax_scale=self.softmax_scale,
                idx_softmax_scale=self.indexer.softmax_scale,
                index_topk=config.index_topk,
                start_pos=start_pos, end_pos=end_pos,
                block_size=config.block_size,
            )
        else:
            q_nope, q_nope_absorbed, q_pe, topk_indices = self.dispatcher.call(
                "fused_q_indexer_score", fused_q_indexer_score,
                qr, normed_x,
                self.wq_b.weight, self.indexer.wq_b.weight,
                self.indexer.weights_proj.weight,
                self.wkv_b.weight,
                freqs_cis, self.indexer.hadamard_matrix,
                self.indexer.k_cache, self.indexer.k_scale_cache,
                eps=config.rms_norm_eps,
                n_heads=self.n_heads, qk_head_dim=self.qk_head_dim,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                idx_n_heads=self.indexer.n_heads,
                idx_head_dim=self.indexer.head_dim,
                rope_head_dim=config.qk_rope_head_dim,
                softmax_scale=self.softmax_scale,
                idx_softmax_scale=self.indexer.softmax_scale,
                index_topk=config.index_topk,
                start_pos=start_pos, end_pos=end_pos,
                block_size=config.block_size,
            )

        if self.dispatcher is None:
            q_pe = fused_q_rope_quant(q_pe, freqs_cis, block_size=config.block_size)
        else:
            q_pe = self.dispatcher.call(
                "fused_q_rope_quant", fused_q_rope_quant,
                q_pe, freqs_cis, block_size=config.block_size,
            )

        kv_c = self.kv_norm(kv_c)
        k_pe_roped = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=True)
        self.kv_cache[:bsz, start_pos:end_pos] = kv_c
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe_roped.squeeze(2)

        if mask is not None:
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv_expanded = self.wkv_b(kv_c)
            kv_expanded = kv_expanded.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe_roped.expand(-1, -1, self.n_heads, -1)], dim=-1)

            scores = torch.einsum("bshd,bthd->bsht", q.float(), k.float()) * self.softmax_scale
            index_mask = torch.full(
                (bsz, seqlen, seqlen), float("-inf"), device=normed_x.device,
            ).scatter_(-1, topk_indices, 0)
            scores = scores + (index_mask + mask).unsqueeze(2)
            scores = scores.softmax(dim=-1).to(v.dtype)
            x = torch.einsum("bsht,bthd->bshd", scores, v)
        else:
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope_absorbed, self.kv_cache[:bsz, :end_pos])
                + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            ) * self.softmax_scale

            index_mask = torch.full(
                (bsz, 1, end_pos), float("-inf"), device=normed_x.device,
            ).scatter_(-1, topk_indices, 0)
            scores = scores + index_mask.unsqueeze(2)
            scores = scores.softmax(dim=-1)

            wkv_b = self.wkv_b.weight.view(self.n_heads, -1, self.kv_lora_rank)
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])

        x = self.wo(x.flatten(2))
        return x"""


# ---------------------------------------------------------------------------
# Recipe definitions (imported by gen_partition.py)
# ---------------------------------------------------------------------------

NVFP4_RECIPES: dict[str, dict] = {
    "fused_ar_rms_qkv_proj": {
        "graph_nodes": ["ar_add_rms", "qkv_a_proj", "indexer_k_proj"],
        "description": "Fused AR+Add+RMS + QKV A Proj + Indexer K",
        "extra_ops": _AR_RMS_QKV_OPS,
        "model_patches": [
            (_BLOCK_FORWARD_OLD, _BLOCK_FORWARD_NEW),
            (_ATTN_FORWARD_OLD, _ATTN_FORWARD_NEW),
        ],
        "requires": [
            "fused_indexer_k_path",
            "fused_q_indexer_score",
            "fused_q_rope_quant",
        ],
    },
    "fused_indexer_k_path": {
        "graph_nodes": [
            "indexer_ln", "indexer_rope", "indexer_quant_fp8", "indexer_cache",
        ],
        "description": "Fused Indexer K: LayerNorm + RoPE + FP8 quant + cache",
        "extra_ops": _INDEXER_K_OPS,
        "model_patches": [],
        "requires": ["fused_ar_rms_qkv_proj"],
    },
    "fused_q_indexer_score": {
        "graph_nodes": [
            "q_rms", "q_b_proj", "indexer_w", "indexer_q_proj",
            "indexer_q_rope", "indexer_q_fp8", "w_uk_t",
            "indexer_w_scale", "indexer_mqa",
        ],
        "description": "Fused Q/Indexer projections + scoring + W_UK_T",
        "extra_ops": _Q_INDEXER_SCORE_OPS,
        "model_patches": [],
        "requires": ["fused_ar_rms_qkv_proj"],
    },
    "fused_q_rope_quant": {
        "graph_nodes": ["q_rope", "cat_q", "q_quant_fp8"],
        "description": "Fused Q RoPE + Cat + Quantize FP8",
        "extra_ops": _Q_ROPE_QUANT_OPS,
        "model_patches": [],
        "requires": ["fused_ar_rms_qkv_proj"],
    },
}
