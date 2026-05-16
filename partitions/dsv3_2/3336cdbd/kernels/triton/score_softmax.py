from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    BACKEND_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    BACKEND_AVAILABLE = False


if BACKEND_AVAILABLE:

    @triton.jit
    def _score_softmax_kernel(
        nope_ptr, rope_ptr, mask_ptr, out_ptr,
        softmax_scale,
        T,
        stride_score_row, stride_score_h, stride_score_t,
        stride_mask_row, stride_mask_t,
        stride_out_row, stride_out_h, stride_out_t,
        BLOCK_T: tl.constexpr,
    ):
        # Grid: (BS_total, H). Each program computes softmax over
        # the time dimension for a single (batch*seq, head) pair.
        # Uses three-pass online softmax: (1) find max, (2) sum exp,
        # (3) normalize and store.
        bs = tl.program_id(0)
        h = tl.program_id(1)

        score_base = bs * stride_score_row + h * stride_score_h
        mask_base = bs * stride_mask_row
        out_base = bs * stride_out_row + h * stride_out_h

        # -- pass 1: find global max across the time dimension ---------
        running_max = float("-inf")
        for block_start in range(0, T, BLOCK_T):
            offs = block_start + tl.arange(0, BLOCK_T)
            valid = offs < T

            nope_val = tl.load(
                nope_ptr + score_base + offs * stride_score_t,
                mask=valid, other=0.0,
            ).to(tl.float32)
            rope_val = tl.load(
                rope_ptr + score_base + offs * stride_score_t,
                mask=valid, other=0.0,
            ).to(tl.float32)
            mask_val = tl.load(
                mask_ptr + mask_base + offs * stride_mask_t,
                mask=valid, other=float("-inf"),
            ).to(tl.float32)

            score = (nope_val + rope_val) * softmax_scale + mask_val
            score = tl.where(valid, score, float("-inf"))
            block_max = tl.max(score, axis=0)
            running_max = tl.maximum(running_max, block_max)

        # -- pass 2: compute sum of exp(score - max) -------------------
        running_sum = 0.0
        for block_start in range(0, T, BLOCK_T):
            offs = block_start + tl.arange(0, BLOCK_T)
            valid = offs < T

            nope_val = tl.load(
                nope_ptr + score_base + offs * stride_score_t,
                mask=valid, other=0.0,
            ).to(tl.float32)
            rope_val = tl.load(
                rope_ptr + score_base + offs * stride_score_t,
                mask=valid, other=0.0,
            ).to(tl.float32)
            mask_val = tl.load(
                mask_ptr + mask_base + offs * stride_mask_t,
                mask=valid, other=float("-inf"),
            ).to(tl.float32)

            score = (nope_val + rope_val) * softmax_scale + mask_val
            exp_val = tl.exp(score - running_max)
            exp_val = tl.where(valid, exp_val, 0.0)
            running_sum += tl.sum(exp_val, axis=0)

        # -- pass 3: write softmax(score) = exp(score-max) / sum -------
        for block_start in range(0, T, BLOCK_T):
            offs = block_start + tl.arange(0, BLOCK_T)
            valid = offs < T

            nope_val = tl.load(
                nope_ptr + score_base + offs * stride_score_t,
                mask=valid, other=0.0,
            ).to(tl.float32)
            rope_val = tl.load(
                rope_ptr + score_base + offs * stride_score_t,
                mask=valid, other=0.0,
            ).to(tl.float32)
            mask_val = tl.load(
                mask_ptr + mask_base + offs * stride_mask_t,
                mask=valid, other=float("-inf"),
            ).to(tl.float32)

            score = (nope_val + rope_val) * softmax_scale + mask_val
            prob = tl.exp(score - running_max) / running_sum
            prob = tl.where(valid, prob, 0.0)

            tl.store(
                out_ptr + out_base + offs * stride_out_t,
                prob,
                mask=valid,
            )


def fused_score_softmax(
    scores_nope: torch.Tensor,
    scores_rope: torch.Tensor,
    index_mask: torch.Tensor,
    *,
    softmax_scale: float,
    fallback,
) -> torch.Tensor:
    del fallback
    if scores_nope.device.type != "cuda":
        raise RuntimeError("Triton fused_score_softmax requires CUDA tensors")

    orig_shape = scores_nope.shape

    if scores_nope.dim() == 4:
        B, S, H, T = scores_nope.shape
    elif scores_nope.dim() == 3:
        H, T = scores_nope.shape[-2], scores_nope.shape[-1]
        B = scores_nope.shape[0]
        S = 1
        scores_nope = scores_nope.unsqueeze(1)
        scores_rope = scores_rope.unsqueeze(1)
    else:
        raise ValueError(f"Expected 3D or 4D scores, got {scores_nope.dim()}D")

    nope_flat = scores_nope.contiguous().view(B * S, H, T)
    rope_flat = scores_rope.contiguous().view(B * S, H, T)

    # The mask broadcasts over H. Flatten to [B*S, T].
    if index_mask.dim() == 4:
        mask_3d = index_mask.squeeze(2)
    elif index_mask.dim() == 3:
        mask_3d = index_mask
    elif index_mask.dim() == 2:
        mask_3d = index_mask.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected mask dim {index_mask.dim()}")

    if mask_3d.shape[1] == 1 and S > 1:
        mask_3d = mask_3d.expand(B, S, T)
    mask_flat = mask_3d.contiguous().view(B * S, T)

    BS_total = B * S
    out = torch.empty(
        BS_total, H, T, device=scores_nope.device, dtype=scores_nope.dtype,
    )

    BLOCK_T = triton.next_power_of_2(min(T, 1024))

    _score_softmax_kernel[(BS_total, H)](
        nope_flat, rope_flat, mask_flat, out,
        softmax_scale,
        T,
        nope_flat.stride(0), nope_flat.stride(1), nope_flat.stride(2),
        mask_flat.stride(0), mask_flat.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_T=BLOCK_T,
        num_warps=8 if BLOCK_T >= 2048 else 4,
    )

    return out.view(orig_shape)
