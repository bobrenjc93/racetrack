"""Attach partition-local kernels to real DeepSeek inference modules.

The synthetic partition models own their own randomly initialized parameters.
For real GSM8K benchmarking we instead keep the checkpoint-backed
``inference.model.Transformer`` intact and let compatible real modules dispatch
through partition kernels from their normal ``forward`` methods. A leaderboard
row is therefore:

    real full model + one partition kernel directory + one backend

Only ops with the same tensor contract as the real inference modules are
patched here. Unsupported partition fusions are left out of the real
leaderboard until they have a real-model adapter.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from racetrack.runtime.dispatch import KernelDispatcher


SUPPORTED_OPS = frozenset({
    "fused_full_topk_indexer",
    "fused_residual_norm",
    "fused_swiglu",
})
BACKENDS = ("torch", "triton", "cutedsl", "helion", "best")
REAL_DISABLED_BACKENDS = {
    "dsv3_2": frozenset({"helion"}),
}


@dataclass(frozen=True)
class RealKernelRow:
    partition_model: str
    partition: str
    backend: str
    kernel_root: Path | None
    ops: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.partition}/{self.backend}"


@dataclass
class PatchStats:
    calls: dict[str, int]
    selected_backends: dict[str, tuple[str, ...]]

    @property
    def total_calls(self) -> int:
        return sum(self.calls.values())

    @property
    def used_partition_kernel(self) -> bool:
        return any(
            backend != "torch"
            for backends in self.selected_backends.values()
            for backend in backends
        )


def discover_real_kernel_rows(
    *,
    partition_model: str = "dsv3_2_nvfp4",
    partition_filter: str = "all",
    backend_filter: str = "all",
) -> list[RealKernelRow]:
    rows = [
        RealKernelRow(
            partition_model=partition_model,
            partition="baseline",
            backend="torch",
            kernel_root=None,
            ops=(),
        )
    ]
    root = Path(__file__).resolve().parents[2] / "partitions" / partition_model
    partitions = sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "kernels").is_dir() and not p.name.startswith("__")
    )
    if partition_filter != "all":
        wanted = {p.strip() for p in partition_filter.split(",") if p.strip()}
        partitions = [p for p in partitions if p.name in wanted]
        if not partitions:
            raise KeyError(f"No partitions matched filter {partition_filter!r}")

    for partition_dir in partitions:
        kernel_root = partition_dir / "kernels"
        concrete_backends = _discover_backends(kernel_root, partition_model)
        backends = list(concrete_backends)
        if backend_filter != "all":
            wanted_backends = {
                _normalize_backend_name(b)
                for b in backend_filter.split(",")
                if b.strip()
            }
            backends = [b for b in backends if b in wanted_backends]
        for backend in backends:
            ops = tuple(sorted(_discover_supported_ops(kernel_root, backend)))
            if not ops and backend != "best":
                continue
            if backend == "best":
                best_ops = set()
                for candidate in concrete_backends:
                    best_ops |= _discover_supported_ops(kernel_root, candidate)
                ops = tuple(sorted(best_ops))
            if not ops:
                continue
            rows.append(
                RealKernelRow(
                    partition_model=partition_model,
                    partition=partition_dir.name,
                    backend=backend,
                    kernel_root=kernel_root,
                    ops=ops,
                )
            )
    return rows


def _normalize_backend_name(backend: str) -> str:
    backend = backend.strip().lower()
    if backend == "cutedl":
        return "cutedsl"
    if backend == "torch_compile":
        return "torch.compile"
    return backend


def _discover_backends(kernel_root: Path, partition_model: str) -> list[str]:
    disabled = REAL_DISABLED_BACKENDS.get(partition_model, frozenset())
    backends = [
        backend for backend in ("triton", "cutedsl", "helion")
        if backend not in disabled and (kernel_root / backend).is_dir()
    ]
    if len(backends) > 1:
        backends.append("best")
    return backends


def _discover_supported_ops(kernel_root: Path, backend: str) -> set[str]:
    if backend == "torch":
        return set()
    dispatcher = KernelDispatcher(kernel_root)
    ops = set()
    for op_name in SUPPORTED_OPS:
        if dispatcher._resolve(backend, op_name) is not None:
            ops.add(op_name)
    return ops


@contextlib.contextmanager
def patch_real_model(
    model: torch.nn.Module,
    row: RealKernelRow,
    *,
    strict_kernel_use: bool = True,
) -> Iterator[PatchStats]:
    if row.backend == "torch" or row.kernel_root is None:
        yield PatchStats(calls={}, selected_backends={})
        return

    dispatcher = _real_model_dispatcher(row)
    stats = PatchStats(
        calls={op_name: 0 for op_name in row.ops},
        selected_backends={},
    )
    originals: list[tuple[torch.nn.Module, str, bool, Any]] = []
    previous_backend = os.environ.get("RACETRACK_KERNEL_BACKEND")
    os.environ["RACETRACK_KERNEL_BACKEND"] = row.backend

    try:
        _patch_modules(model, row, dispatcher, stats, originals)
        yield stats
        if row.backend == "best":
            stats.selected_backends.update(
                {
                    op_name: tuple(sorted(backends))
                    for op_name, backends in dispatcher._best_ops.items()
                }
            )
        else:
            stats.selected_backends.update(
                {op_name: (row.backend,) for op_name, calls in stats.calls.items() if calls}
            )
        if strict_kernel_use and stats.total_calls == 0:
            raise RuntimeError(f"{row.label} did not exercise any compatible real-model kernels")
        if strict_kernel_use and not stats.used_partition_kernel:
            raise RuntimeError(f"{row.label} did not use any non-torch partition kernel")
    finally:
        for module, name, existed, original in reversed(originals):
            if existed:
                setattr(module, name, original)
            else:
                delattr(module, name)
        if previous_backend is None:
            os.environ.pop("RACETRACK_KERNEL_BACKEND", None)
        else:
            os.environ["RACETRACK_KERNEL_BACKEND"] = previous_backend


def _patch_modules(
    model: torch.nn.Module,
    row: RealKernelRow,
    dispatcher: Any,
    stats: PatchStats,
    originals: list[tuple[torch.nn.Module, str, bool, Any]],
) -> None:
    from inference import model as real_model

    for module in model.modules():
        if "fused_full_topk_indexer" in row.ops and isinstance(module, real_model.MLA):
            _attach_mla_full_topk_forward(module, originals)
        if "fused_full_topk_indexer" in row.ops and isinstance(module, real_model.Indexer):
            _attach_indexer_kernel(module, dispatcher, stats, originals)
        if "fused_residual_norm" in row.ops and isinstance(module, real_model.RMSNorm):
            _attach_kernel(module, dispatcher, stats, originals)
        if "fused_swiglu" in row.ops and isinstance(module, (real_model.MLP, real_model.Expert)):
            _attach_kernel(module, dispatcher, stats, originals)


def _attach_kernel(
    module: torch.nn.Module,
    dispatcher: Any,
    stats: PatchStats,
    originals: list[tuple[torch.nn.Module, str, bool, Any]],
) -> None:
    _set_attr(module, "kernel_dispatcher", dispatcher, originals)
    _set_attr(module, "kernel_stats", stats, originals)


def _attach_indexer_kernel(
    module: torch.nn.Module,
    dispatcher: Any,
    stats: PatchStats,
    originals: list[tuple[torch.nn.Module, str, bool, Any]],
) -> None:
    op_name = "fused_full_topk_indexer"
    original_forward = module.forward
    _set_attr(module, "_racetrack_pending_full_topk", [], originals)

    def forward(
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        stats.calls[op_name] = stats.calls.get(op_name, 0) + 1

        def fallback(
            indexer: torch.nn.Module,
            hidden: torch.Tensor,
            q_residual: torch.Tensor,
            pos: int,
            freqs: torch.Tensor,
            attn_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            del indexer
            return original_forward(hidden, q_residual, pos, freqs, attn_mask)

        return dispatcher.call(
            op_name,
            fallback,
            module,
            x,
            qr,
            start_pos,
            freqs_cis,
            mask,
        )

    _set_attr(module, "forward", forward, originals)


def _attach_mla_full_topk_forward(
    module: torch.nn.Module,
    originals: list[tuple[torch.nn.Module, str, bool, Any]],
) -> None:
    original_forward = module.forward

    def forward(
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if end_pos > module.indexer.index_topk:
            return original_forward(x, start_pos, freqs_cis, mask)
        return _mla_full_topk_forward(module, x, start_pos, freqs_cis, mask)

    _set_attr(module, "forward", forward, originals)


def _mla_full_topk_forward(
    module: torch.nn.Module,
    x: torch.Tensor,
    start_pos: int,
    freqs_cis: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    from inference import model as real_model

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
        kv = kv.view(
            bsz,
            seqlen,
            module.n_local_heads,
            module.qk_nope_head_dim + module.v_head_dim,
        )
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


def _set_attr(
    module: torch.nn.Module,
    name: str,
    value: Any,
    originals: list[tuple[torch.nn.Module, str, bool, Any]],
) -> None:
    existed = hasattr(module, name)
    original = getattr(module, name, None)
    originals.append((module, name, existed, original))
    setattr(module, name, value)


def _real_model_dispatcher(row: RealKernelRow) -> Any:
    dispatcher = KernelDispatcher(row.kernel_root)
    disabled = REAL_DISABLED_BACKENDS.get(row.partition_model, frozenset())
    if disabled:
        dispatcher.BACKENDS = tuple(
            backend for backend in dispatcher.BACKENDS
            if backend not in disabled
        )
        dispatcher._best_fast_path = {
            op_name: backend
            for op_name, backend in dispatcher._best_fast_path.items()
            if backend not in disabled
        }
    if row.partition_model == "dsv3_2":
        return _Dsv3RealContractDispatcher(dispatcher)
    return dispatcher


class _Dsv3RealContractDispatcher:
    """Adapts legacy dsv3_2 partition kernels to inference.model contracts."""

    def __init__(self, dispatcher: KernelDispatcher) -> None:
        self._dispatcher = dispatcher

    @property
    def _best_ops(self) -> dict[str, set[str]]:
        return self._dispatcher._best_ops

    def call(self, op_name: str, fallback, *args: Any, **kwargs: Any) -> Any:
        if op_name == "fused_residual_norm":
            return self._call_residual_norm(fallback, *args, **kwargs)
        if op_name == "fused_swiglu":
            return self._call_swiglu(fallback, *args, **kwargs)
        return self._dispatcher.call(op_name, fallback, *args, **kwargs)

    def _call_residual_norm(
        self,
        fallback,
        update: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = update.shape
        cols = shape[-1]
        update_flat = update.contiguous().view(-1, cols)
        residual_flat = residual.contiguous().view(-1, cols)

        def legacy_fallback(
            legacy_residual: torch.Tensor,
            legacy_update: torch.Tensor,
            legacy_weight: torch.Tensor,
            *,
            eps: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            normed, hidden = fallback(
                legacy_update.view(shape),
                legacy_residual.view(shape),
                legacy_weight,
                eps=eps,
            )
            return hidden.contiguous().view(-1, cols), normed.contiguous().view(-1, cols)

        hidden, normed = self._dispatcher.call(
            "fused_residual_norm",
            legacy_fallback,
            residual_flat,
            update_flat,
            weight,
            eps=eps,
        )
        return normed.view(shape), hidden.view(shape)

    def _call_swiglu(
        self,
        fallback,
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        shape = gate.shape
        cols = shape[-1]
        gate_flat = gate.contiguous().view(-1, cols)
        up_flat = up.contiguous().view(-1, cols)

        def legacy_fallback(
            legacy_gate: torch.Tensor,
            legacy_up: torch.Tensor,
        ) -> torch.Tensor:
            return fallback(
                legacy_gate.view(shape),
                legacy_up.view(shape),
            ).contiguous().view(-1, cols)

        return self._dispatcher.call(
            "fused_swiglu",
            legacy_fallback,
            gate_flat,
            up_flat,
        ).view(shape)
