"""FP8 block-wise GEMM kernel for DeepSeek-V3.2 inference.

Uses a Triton kernel that fuses block-wise dequantization into the matmul,
leveraging H100 FP8 tensor cores. JIT-compiled kernels are cached in
~/.triton/cache across runs.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

block_size = 128

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _fp8_gemm_kernel(
        A_ptr, B_ptr, C_ptr, A_s_ptr, B_s_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bn, stride_bk,
        stride_cm, stride_cn,
        stride_as_m, stride_as_k, stride_bs_n, stride_bs_k,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_idx in range(tl.cdiv(K, BLOCK_K)):
            k_start = k_idx * BLOCK_K
            offs_k = k_start + tl.arange(0, BLOCK_K)

            a = tl.load(
                A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
            )
            b = tl.load(
                B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0,
            )
            a_scale = tl.load(
                A_s_ptr + offs_m * stride_as_m + k_idx * stride_as_k,
                mask=offs_m < M, other=1.0,
            )
            b_n_block = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) // 128
            b_scale = tl.load(
                B_s_ptr + b_n_block * stride_bs_n + k_idx * stride_bs_k,
                mask=offs_n < N, other=1.0,
            )

            acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32) * a_scale[:, None] * b_scale[None, :]

        c = acc.to(tl.bfloat16)
        tl.store(
            C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    N = x.size(-1)
    if N % block_size != 0:
        return x.to(torch.float8_e4m3fn), torch.ones(
            *x.shape[:-1], 1, dtype=torch.float32, device=x.device,
        )
    n_groups = N // block_size
    shape = x.shape
    x_flat = x.float().reshape(-1, n_groups, block_size)
    amax = x_flat.abs().amax(dim=-1).clamp(min=1e-4)
    scale = amax / 448.0
    x_scaled = x_flat / scale.unsqueeze(-1)
    y = x_scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(shape)
    s = scale.view(*shape[:-1], n_groups)
    return y, s


def fp8_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor
) -> torch.Tensor:
    if not (_TRITON_AVAILABLE and a.is_cuda and b.dtype == torch.float8_e4m3fn):
        return F.linear(a.to(torch.bfloat16), _block_dequant(b, b_s))

    orig_shape = a.shape
    if a.dtype != torch.float8_e4m3fn:
        a, a_s = act_quant(a, block_size)
    if a.dim() > 2:
        a = a.reshape(-1, a.shape[-1])
        a_s = a_s.reshape(-1, a_s.shape[-1])

    M, K = a.shape
    N = b.shape[0]
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    BLOCK_M, BLOCK_N = 64, 128
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _fp8_gemm_kernel[grid](
        a, b, c, a_s, b_s,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        a_s.stride(0), a_s.stride(1), b_s.stride(0), b_s.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=128, GROUP_SIZE_M=8,
        num_warps=4, num_stages=3,
    )
    return c.view(*orig_shape[:-1], N)


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
