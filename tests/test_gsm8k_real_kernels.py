from __future__ import annotations

import textwrap

import torch

from benchmarks.gsm8k.real_kernels import (
    RealKernelRow,
    discover_real_kernel_rows,
    patch_real_model,
)
from benchmarks.gsm8k.real_bench import ExampleResult, _render_markdown, _row_result
from benchmarks.gsm8k.hf_model_loader import _slice_for_rank, run_post_load_transforms
from racetrack.partition_spec import FusedOp, PartitionSpec


def _make_spec(
    model: str,
    partition: str,
    ops: list[dict],
    kernel_root=None,
) -> PartitionSpec:
    return PartitionSpec(
        model=model,
        partition_hash=partition,
        notes="test",
        graph_nodes={},
        fused_ops=tuple(FusedOp(name=o["name"], kind=o["kind"]) for o in ops),
    )


def test_real_kernel_patcher_uses_real_module_weights(tmp_path) -> None:
    kernel_dir = tmp_path / "kernels" / "triton"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "ops.py").write_text(
        textwrap.dedent(
            """
            BACKEND_AVAILABLE = True

            def fused_swiglu(gate, up, *, fallback):
                return fallback(gate, up)

            def fused_residual_norm(x, residual, weight, *, eps, fallback):
                return fallback(x, residual, weight, eps=eps)
            """
        )
    )

    from racetrack.models.deepseek import MLP, RMSNorm

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = RMSNorm(8)
            self.mlp = MLP(8, 16)

        def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x, residual = self.norm(x, residual)
            return self.mlp(x), residual

    model = Tiny().float().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.02)
        model.norm.weight.fill_(1.0)
    x = torch.randn(2, 3, 8)
    residual = torch.randn(2, 3, 8)
    expected = model(x, residual)

    spec = _make_spec("dsv3_2_nvfp4", "test", [
        {"name": "fused_residual_norm", "kind": "fx_pattern"},
        {"name": "fused_swiglu", "kind": "fx_pattern"},
    ])
    row = RealKernelRow(
        partition_model="dsv3_2_nvfp4",
        partition="test",
        backend="triton",
        kernel_root=tmp_path / "kernels",
        ops=("fused_residual_norm", "fused_swiglu"),
        spec=spec,
    )
    with patch_real_model(model, row) as stats:
        actual = model(x, residual)
        assert hasattr(model.norm, "kernel_dispatcher")
        assert hasattr(model.mlp, "kernel_dispatcher")

    assert stats.calls == {"fused_residual_norm": 1, "fused_swiglu": 1}
    assert stats.used_partition_kernel
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert not hasattr(model.norm, "kernel_dispatcher")
    assert not hasattr(model.norm, "kernel_stats")
    assert not hasattr(model.mlp, "kernel_dispatcher")
    assert not hasattr(model.mlp, "kernel_stats")


def test_dsv3_2_residual_norm_adapter_handles_legacy_kernel_contract(tmp_path) -> None:
    kernel_dir = tmp_path / "kernels" / "triton"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "ops.py").write_text(
        textwrap.dedent(
            """
            BACKEND_AVAILABLE = True

            def fused_residual_norm(residual, update, weight, *, eps, fallback):
                assert residual.dim() == 2
                assert update.dim() == 2
                hidden, normed = fallback(residual, update, weight, eps=eps)
                return hidden, normed
            """
        )
    )

    from racetrack.models.deepseek import RMSNorm

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = RMSNorm(8)

        def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.norm(x, residual)

    model = Tiny().float().eval()
    with torch.no_grad():
        model.norm.weight.normal_(mean=1.0, std=0.02)
    x = torch.randn(2, 3, 8)
    residual = torch.randn(2, 3, 8)
    expected = model(x, residual)

    spec = _make_spec("dsv3_2", "test", [
        {"name": "fused_residual_norm", "kind": "fx_pattern"},
    ])
    row = RealKernelRow(
        partition_model="dsv3_2",
        partition="test",
        backend="triton",
        kernel_root=tmp_path / "kernels",
        ops=("fused_residual_norm",),
        spec=spec,
    )

    with patch_real_model(model, row) as stats:
        actual = model(x, residual)

    assert stats.calls == {"fused_residual_norm": 1}
    assert stats.used_partition_kernel
    assert actual[0].shape == x.shape
    assert actual[1].shape == x.shape
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_dsv3_2_swiglu_adapter_handles_legacy_kernel_contract(tmp_path) -> None:
    kernel_dir = tmp_path / "kernels" / "triton"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "ops.py").write_text(
        textwrap.dedent(
            """
            BACKEND_AVAILABLE = True

            def fused_swiglu(gate, up, *, fallback):
                assert gate.dim() == 2
                assert up.dim() == 2
                return fallback(gate, up)
            """
        )
    )

    from racetrack.models.deepseek import MLP

    model = MLP(8, 16).float().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    x = torch.randn(2, 3, 8)
    expected = model(x)

    spec = _make_spec("dsv3_2", "test", [
        {"name": "fused_swiglu", "kind": "fx_pattern"},
    ])
    row = RealKernelRow(
        partition_model="dsv3_2",
        partition="test",
        backend="triton",
        kernel_root=tmp_path / "kernels",
        ops=("fused_swiglu",),
        spec=spec,
    )

    with patch_real_model(model, row) as stats:
        actual = model(x)

    assert stats.calls == {"fused_swiglu": 1}
    assert stats.used_partition_kernel
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_dsv3_2_real_rows_skip_helion_until_full_model_configs_exist() -> None:
    rows = discover_real_kernel_rows(
        partition_model="dsv3_2",
        partition_filter="3336cdbd",
        backend_filter="all",
    )

    assert rows
    assert all(row.backend != "helion" for row in rows)
    partition_rows = [r for r in rows if r.partition == "3336cdbd" and r.backend != "torch" and r.backend != "torch.compile"]
    assert partition_rows
    for row in partition_rows:
        assert row.spec is not None


def test_dsv3_2_nvfp4_real_rows_include_full_topk_indexer() -> None:
    rows = discover_real_kernel_rows(
        partition_model="dsv3_2_nvfp4",
        partition_filter="cd91301b",
        backend_filter="triton",
    )

    row = next(row for row in rows if row.partition == "cd91301b")
    assert row.spec is not None
    assert "fused_full_topk_indexer" in row.ops
    assert "fused_residual_norm" in row.ops
    assert "fused_swiglu" in row.ops


def test_dsv3_2_best_ignores_cached_disabled_backend(tmp_path) -> None:
    triton_dir = tmp_path / "kernels" / "triton"
    helion_dir = tmp_path / "kernels" / "helion"
    triton_dir.mkdir(parents=True)
    helion_dir.mkdir(parents=True)
    (tmp_path / "kernels" / "best.json").write_text('{"fused_swiglu": "helion"}\n')
    (triton_dir / "ops.py").write_text(
        textwrap.dedent(
            """
            BACKEND_AVAILABLE = True

            def fused_swiglu(gate, up, *, fallback):
                return fallback(gate, up)
            """
        )
    )
    (helion_dir / "ops.py").write_text(
        textwrap.dedent(
            """
            BACKEND_AVAILABLE = True

            def fused_swiglu(gate, up, *, fallback):
                raise RuntimeError("helion should not run")
            """
        )
    )

    from racetrack.models.deepseek import MLP

    model = MLP(8, 16).float().eval()
    x = torch.randn(2, 3, 8)

    spec = _make_spec("dsv3_2", "test", [
        {"name": "fused_swiglu", "kind": "fx_pattern"},
    ])
    row = RealKernelRow(
        partition_model="dsv3_2",
        partition="test",
        backend="best",
        kernel_root=tmp_path / "kernels",
        ops=("fused_swiglu",),
        spec=spec,
    )

    with patch_real_model(model, row, strict_kernel_use=False) as stats:
        model(x)

    selected = {
        backend
        for backends in stats.selected_backends.values()
        for backend in backends
    }
    assert "helion" not in selected


def test_real_kernel_patcher_can_route_indexer_forward(tmp_path) -> None:
    kernel_dir = tmp_path / "kernels" / "triton"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "ops.py").write_text(
        textwrap.dedent(
            """
            import torch

            BACKEND_AVAILABLE = True

            def fused_full_topk_indexer(indexer, x, qr, start_pos, freqs_cis, mask, *, fallback):
                del fallback, indexer, qr, freqs_cis, mask
                end_pos = start_pos + x.shape[1]
                return torch.arange(end_pos, device=x.device).view(1, 1, end_pos).expand(
                    x.shape[0], x.shape[1], end_pos,
                )
            """
        )
    )

    from racetrack.models import deepseek as real_model

    class TinyIndexer(real_model.Indexer):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.index_topk = 64

        def forward(self, x, qr, start_pos, freqs_cis, mask):
            raise AssertionError("fallback should not run")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.indexer = TinyIndexer()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.indexer(x, x, 2, torch.empty(3, 2), None)

    model = Tiny()
    spec = _make_spec("dsv3_2", "test", [
        {"name": "fused_full_topk_indexer", "kind": "pre_trace"},
    ])
    row = RealKernelRow(
        partition_model="dsv3_2",
        partition="test",
        backend="triton",
        kernel_root=tmp_path / "kernels",
        ops=("fused_full_topk_indexer",),
        spec=spec,
    )

    with patch_real_model(model, row) as stats:
        topk = model(torch.randn(1, 3, 4))

    assert topk.shape == (1, 3, 5)
    assert torch.equal(topk[0, 0], torch.arange(5))


def test_real_kernel_patcher_can_route_moe_forward(tmp_path) -> None:
    kernel_dir = tmp_path / "kernels" / "triton"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "ops.py").write_text(
        textwrap.dedent(
            """
            BACKEND_AVAILABLE = True

            def fused_single_token_moe(moe, x, *, fallback):
                del fallback, moe
                return x + 1
            """
        )
    )

    from racetrack.models import deepseek as real_model

    class TinyMoE(real_model.MoE):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise AssertionError("fallback should not run")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moe = TinyMoE()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.moe(x)

    model = Tiny()
    spec = _make_spec("dsv3_2_nvfp4", "test", [
        {"name": "fused_single_token_moe", "kind": "pre_trace"},
    ])
    row = RealKernelRow(
        partition_model="dsv3_2_nvfp4",
        partition="test",
        backend="triton",
        kernel_root=tmp_path / "kernels",
        ops=("fused_single_token_moe",),
        spec=spec,
    )

    x = torch.randn(1, 1, 4)
    with patch_real_model(model, row) as stats:
        actual = model(x)

    assert torch.equal(actual, x + 1)


def test_real_kernel_patcher_can_route_mlp_gate_up_projection(tmp_path) -> None:
    kernel_dir = tmp_path / "kernels" / "triton"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "ops.py").write_text(
        textwrap.dedent(
            """
            BACKEND_AVAILABLE = True

            def fused_mlp_gate_up_proj(
                x, w1_weight, w1_scale, w3_weight, w3_scale, *, scale_fmt, fallback,
            ):
                return fallback(
                    x, w1_weight, w1_scale, w3_weight, w3_scale, scale_fmt=scale_fmt,
                )
            """
        )
    )

    from racetrack.models.deepseek import MLP

    model = MLP(8, 16).float().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    x = torch.randn(2, 3, 8)
    expected = model(x)

    spec = _make_spec("dsv3_2", "test", [
        {"name": "fused_mlp_gate_up_proj", "kind": "module_patch"},
    ])
    row = RealKernelRow(
        partition_model="dsv3_2",
        partition="test",
        backend="triton",
        kernel_root=tmp_path / "kernels",
        ops=("fused_mlp_gate_up_proj",),
        spec=spec,
    )

    with patch_real_model(model, row) as stats:
        actual = model(x)

    assert torch.equal(actual, expected)


def test_hf_loader_slices_rows_and_columns_by_target_shape() -> None:
    value = torch.arange(24).view(6, 4)

    row_target = torch.empty(2, 4)
    assert torch.equal(
        _slice_for_rank(value, row_target, "row", rank=2),
        value[4:6],
    )

    col_target = torch.empty(6, 2)
    assert torch.equal(
        _slice_for_rank(value, col_target, "col", rank=1),
        value[:, 2:4],
    )


def test_real_row_validation_uses_extracted_answers_not_exact_tokens() -> None:
    row = RealKernelRow(
        partition_model="dsv3_2_nvfp4",
        partition="test",
        backend="triton",
        kernel_root=None,
        ops=("fused_swiglu",),
    )
    baseline = [
        ExampleResult(
            completion_tokens=(1, 2, 3),
            predicted=42.0,
            ground_truth=42.0,
            correct=True,
        )
    ]
    outputs = [
        ExampleResult(
            completion_tokens=(4, 5, 6),
            predicted=42.0,
            ground_truth=42.0,
            correct=True,
        )
    ]

    result = _row_result(row, outputs, 1.0, baseline, {"fused_swiglu": 1})

    assert result.validation
    assert result.accuracy_pct == 100.0
    assert result.answer_match == 1
    assert result.token_match == 0
    assert result.max_abs_diff == 0.0


def test_real_report_keeps_legacy_leaderboard_schema() -> None:
    report = {
        "model": "test-model",
        "partition_model": "dsv3_2_nvfp4",
        "hardware": {
            "gpu": "test-gpu",
            "gpu_count": 8,
            "cuda": "test-cuda",
            "torch": "test-torch",
        },
        "timestamp": "2026-01-01T00:00:00+00:00",
        "samples": 1,
        "max_new_tokens": 16,
        "rows": [
            {
                "partition": "baseline",
                "backend": "torch",
                "ops": [],
                "mean_ms": 10.0,
                "total_ms": 10.0,
                "accuracy_pct": 100.0,
                "correct": 1,
                "total": 1,
                "validation": True,
                "answer_match": 1,
                "token_match": 1,
                "max_abs_diff": 0.0,
                "calls": {},
                "selected_backends": {},
            },
            {
                "partition": "test",
                "backend": "best",
                "ops": ["fused_swiglu"],
                "mean_ms": 5.0,
                "total_ms": 5.0,
                "accuracy_pct": 100.0,
                "correct": 1,
                "total": 1,
                "validation": True,
                "answer_match": 1,
                "token_match": 0,
                "max_abs_diff": 0.0,
                "calls": {"fused_swiglu": 1},
                "selected_backends": {"fused_swiglu": ["triton"]},
            },
        ],
    }

    markdown = _render_markdown(report, "8xh100")

    assert "| # | partition | backend | total (ms) | vs baseline | validation | max diff |" in markdown
    assert "answer match" not in markdown
    assert "token match" not in markdown
    assert "| 1 | test | best (fused_swiglu=triton) | 5.0 | 2.000x | pass | 0.000e+00 |" in markdown


def test_post_load_transform_hooks_run_once_per_module() -> None:
    class Hooked(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.called: list[str] = []

        def rebuild_derived_weights(self) -> None:
            self.called.append("rebuild")

        def fuse_indexer_weights(self) -> None:
            self.called.append("fuse")

    model = torch.nn.Sequential(Hooked())

    called = run_post_load_transforms(model)

    assert called == ["0.rebuild_derived_weights"]
    assert model[0].called == ["rebuild"]


def test_partition_spec_loads_and_discovers() -> None:
    from racetrack.partition_spec import discover_partitions, load_spec

    spec = load_spec("dsv3_2", "3336cdbd")
    assert spec.model == "dsv3_2"
    assert spec.partition_hash == "3336cdbd"
    assert len(spec.fused_ops) == 6
    assert len(spec.fx_ops) == 3
    assert len(spec.pre_trace_ops) == 2
    assert len(spec.module_patch_ops) == 1

    all_specs = discover_partitions("dsv3_2")
    assert len(all_specs) == 5
    hashes = {s.partition_hash for s in all_specs}
    assert "3336cdbd" in hashes
    assert "a1f6d7e2" in hashes
