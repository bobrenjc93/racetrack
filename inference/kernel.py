"""Pure-torch replacements for the FP8 tilelang kernels.

Weights stay in FP8 on GPU. Block-wise dequantization happens on every
matmul call. This is slower than tilelang but correct at all sequence
lengths and doesn't require extra GPU memory for BF16 weight copies.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

block_size = 128


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = x.size(-1)
    n_groups = max(N // block_size, 1)
    return x.contiguous(), torch.ones(
        *x.shape[:-1], n_groups, dtype=torch.float32, device=x.device,
    )


def fp8_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor
) -> torch.Tensor:
    K = a.size(-1)
    N = b.size(0)
    a_shape = a.shape

    b_dequant = _block_dequant(b, b_s)
    return F.linear(a.to(torch.bfloat16), b_dequant).view(*a_shape[:-1], N)


def _block_dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    shape = weight.shape
    bs = block_size
    out_f, in_f = shape
    pad_out = (bs - out_f % bs) % bs
    pad_in = (bs - in_f % bs) % bs
    w = weight.float()
    if pad_out > 0 or pad_in > 0:
        w = F.pad(w, (0, pad_in, 0, pad_out))
    po, pi = w.shape
    w = w.view(po // bs, bs, pi // bs, bs).transpose(1, 2).contiguous()
    w = (w.view(-1, bs * bs) * scale.view(-1, 1).float())
    w = w.view(po // bs, pi // bs, bs, bs).transpose(1, 2).contiguous().view(po, pi)
    return w[:out_f, :in_f].to(torch.bfloat16)


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    b, m, h, d = q.shape
    q_f = q.float()
    k_f = k.float()
    logits = torch.einsum("bmhd,bnd->bmhn", q_f, k_f)
    logits = torch.relu(logits) * q_s.unsqueeze(-1)
    logits_sum = logits.sum(dim=2)
    return logits_sum * k_s.unsqueeze(1)
