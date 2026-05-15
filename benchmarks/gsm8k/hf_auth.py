"""Hugging Face token helpers for GSM8K benchmark entry points."""

from __future__ import annotations

import os
from pathlib import Path

TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "hf_token")


def _clean_token(value: str) -> str:
    return value.strip().strip('"').strip("'")


def load_hf_token(
    cli_token: str | None = None,
    *,
    env_path: Path | None = None,
) -> str | None:
    if cli_token:
        token = _clean_token(cli_token)
        if token:
            return token

    for key in TOKEN_KEYS:
        value = os.getenv(key)
        if value:
            token = _clean_token(value)
            if token:
                return token

    path = env_path or Path.home() / ".env"
    if not path.exists():
        return None

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() in TOKEN_KEYS:
            token = _clean_token(value)
            if token:
                return token
    return None


def require_hf_token(
    cli_token: str | None = None,
    *,
    env_path: Path | None = None,
    purpose: str = "GSM8K benchmark",
) -> str:
    token = load_hf_token(cli_token, env_path=env_path)
    if token is None:
        raise ValueError(
            f"{purpose} requires a Hugging Face token. Pass --hf-token, set "
            "HF_TOKEN, or add hf_token=... to ~/.env. Use --dummy-weights "
            "only for local synthetic optimization runs."
        )
    return token
