from __future__ import annotations

import torch
import pytest

from racetrack.bench import parse_args, run


def test_cpu_smoke_all_models_all_backends() -> None:
    args = parse_args(
        [
            "--model",
            "all",
            "--partition",
            "all",
            "--kernel-filter",
            "all",
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
    assert {result.model for result in results} == {"dsv3_2", "dsv4", "ds"}


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
    if model.backend_status["cutedsl"] == "missing":
        with pytest.raises(RuntimeError, match="No available cutedsl kernel"):
            model(input_ids)
    else:
        out = model(input_ids)
        assert out.shape == (4, model.config.vocab_size)
