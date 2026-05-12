from __future__ import annotations

from pathlib import Path

from racetrack.runtime import FlattenedDeepSeekModel
from racetrack.runtime.modeling import model_config

MODEL_NAME = "dsv4"
PARTITION_HASH = "b49c2a81"
PARTITION_NOTES = (
    "Wraps DSV4 hyper-compressed residual head and MLA norm/RoPE staging in "
    "backend-selectable kernel calls, following vLLM PR 40860's V4 layer shape."
)


def build_model(**overrides) -> FlattenedDeepSeekModel:
    config = model_config(MODEL_NAME).for_benchmark(**overrides)
    return FlattenedDeepSeekModel(config, partition_root=Path(__file__).parent)
