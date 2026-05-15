from __future__ import annotations

import math

import pytest
import torch

from benchmarks.gsm8k.bench import Result, _validate_output, pick_winner
from benchmarks.gsm8k.hf_auth import load_hf_token, require_hf_token


def _result(partition: str, backend: str, mean_ms: float, *, ok: bool) -> Result:
    diff = 0.0 if ok else math.inf
    return Result(
        model="dsv3_2",
        partition=partition,
        backend=backend,
        backend_status="native",
        case="prefill_512",
        tokens=512,
        device="cpu",
        dtype="float32",
        mean_ms=mean_ms,
        min_ms=mean_ms,
        max_ms=mean_ms,
        tokens_per_second=512 / (mean_ms / 1000.0),
        max_abs_diff=diff,
        max_rel_diff=diff,
        ok=ok,
    )


def test_validate_output_rejects_shape_and_nonfinite_mismatches() -> None:
    baseline = torch.tensor([[1.0, float("nan")]])

    assert _validate_output(
        torch.tensor([[1.0, float("nan")]]),
        baseline,
        atol=0.0,
        rtol=0.0,
    ) == (0.0, 0.0, True)

    _, _, finite_ok = _validate_output(
        torch.tensor([[1.0, 2.0]]),
        baseline,
        atol=0.0,
        rtol=0.0,
    )
    assert not finite_ok

    _, _, shape_ok = _validate_output(
        torch.ones(1, 1),
        baseline,
        atol=0.0,
        rtol=0.0,
    )
    assert not shape_ok


def test_pick_winner_prefers_valid_outputs_over_fast_invalid_outputs() -> None:
    report = pick_winner(
        [
            _result("baseline", "torch", 10.0, ok=True),
            _result("fastbad", "torch", 1.0, ok=False),
            _result("valid", "torch", 9.0, ok=True),
        ]
    )

    assert report["winner"]["partition"] == "valid"
    assert report["winner"]["backend"] == "torch"
    assert report["winner"]["ok"] is True


def test_hf_token_loader_reads_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("hf_token", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("  hf_token = 'test-token'  \n")

    assert load_hf_token(env_path=env_path) == "test-token"


def test_hf_token_required_for_non_dummy_reports(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("hf_token", raising=False)

    with pytest.raises(ValueError, match="requires a Hugging Face token"):
        require_hf_token(env_path=tmp_path / "missing.env")
