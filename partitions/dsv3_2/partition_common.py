from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


_BASE_PARTITION = importlib.import_module("partitions.dsv3_2.3336cdbd.model")
BASE_CONFIG = _BASE_PARTITION.DSV3_2_CONFIG
FlattenedDeepSeekModel = _BASE_PARTITION.FlattenedDeepSeekModel


def build_partition_model(
    partition_root: str | Path,
    partition_overrides: dict[str, Any] | None = None,
    **benchmark_overrides: Any,
) -> FlattenedDeepSeekModel:
    overrides = {
        **(partition_overrides or {}),
        **benchmark_overrides,
    }
    config = BASE_CONFIG.for_benchmark(**overrides)
    return FlattenedDeepSeekModel(config, partition_root=partition_root)
