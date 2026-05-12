from __future__ import annotations

from pathlib import Path

from racetrack.runtime import FlattenedDeepSeekModel
from racetrack.runtime.modeling import model_config

MODEL_NAME = "ds"
PARTITION_HASH = "c7d9e510"
PARTITION_NOTES = (
    "Generic DeepSeek MLA/MoE partition for comparing backend plumbing with "
    "smaller shapes than DSV3.2 and DSV4."
)


def build_model(**overrides) -> FlattenedDeepSeekModel:
    config = model_config(MODEL_NAME).for_benchmark(**overrides)
    return FlattenedDeepSeekModel(config, partition_root=Path(__file__).parent)
