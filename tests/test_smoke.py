from __future__ import annotations

import torch
import pytest

from racetrack.bench import CONCRETE_KERNEL_BACKENDS, parse_args, run


def test_cpu_smoke_dsv3_2_torch() -> None:
    args = parse_args(
        [
            "--model",
            "dsv3_2",
            "--partition",
            "baseline",
            "--kernel-filter",
            "torch",
            "--benchmark",
            "smoke",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--warmup",
            "0",
            "--repeat",
            "1",
        ]
    )
    results = run(args)
    assert results
    assert all(result.ok for result in results)
    assert {result.model for result in results} == {"dsv3_2"}
    assert {result.backend for result in results} == {"torch"}


def test_missing_backend_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from racetrack.runtime.dispatch import KernelDispatcher

    monkeypatch.setenv("RACETRACK_KERNEL_BACKEND", "cutedsl")
    dispatcher = KernelDispatcher()
    with pytest.raises(RuntimeError, match="No available cutedsl kernel"):
        dispatcher.call("fused_norm_rope", lambda: None)


def test_backend_env_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RACETRACK_KERNEL_BACKEND", "cutedl")
    from partitions.dsv3_2.a1f6d7e2.model import build_model

    model = build_model().eval()
    input_ids = torch.arange(4, dtype=torch.long)
    assert model.backend_status["cutedsl"] in {"native", "missing"}
    expected = (
        "CUTEDSL kernels require CUDA tensors"
        if model.backend_status["cutedsl"] == "native"
        else "No available cutedsl kernel"
    )
    with pytest.raises(RuntimeError, match=expected):
        model(input_ids)


def test_all_uses_only_implemented_backends() -> None:
    assert CONCRETE_KERNEL_BACKENDS == ("triton", "cutedsl", "helion")


def test_only_one_model_is_supported() -> None:
    from racetrack.bench import MODELS
    from racetrack.runtime.modeling import model_config

    assert MODELS == ("dsv3_2",)
    with pytest.raises(KeyError, match="Unknown model config"):
        model_config("not_dsv3_2")
