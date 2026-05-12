from __future__ import annotations

import os

import torch

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


def test_backend_env_alias() -> None:
    os.environ["RACETRACK_KERNEL_BACKEND"] = "cutedl"
    from partitions.dsv3_2.a1f6d7e2.model import build_model

    model = build_model().eval()
    input_ids = torch.arange(4, dtype=torch.long)
    out = model(input_ids)
    assert out.shape == (4, model.config.vocab_size)
