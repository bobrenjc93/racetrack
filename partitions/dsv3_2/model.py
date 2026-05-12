from __future__ import annotations

from racetrack.runtime import FlattenedDeepSeekModel
from racetrack.runtime.modeling import model_config

MODEL_NAME = "dsv3_2"


def build_model(**overrides) -> FlattenedDeepSeekModel:
    config = model_config(MODEL_NAME).for_benchmark(**overrides)
    return FlattenedDeepSeekModel(config)
