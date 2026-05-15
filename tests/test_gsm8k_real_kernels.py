from __future__ import annotations

import textwrap

import torch

from benchmarks.gsm8k.real_kernels import RealKernelRow, patch_real_model
from benchmarks.gsm8k.real_bench import ExampleResult, _row_result
from benchmarks.gsm8k.hf_model_loader import _slice_for_rank, run_post_load_transforms


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

    from inference.model import MLP, RMSNorm

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

    row = RealKernelRow(
        partition_model="dsv3_2_nvfp4",
        partition="test",
        backend="triton",
        kernel_root=tmp_path / "kernels",
        ops=("fused_residual_norm", "fused_swiglu"),
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
