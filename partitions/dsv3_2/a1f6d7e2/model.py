from __future__ import annotations

from pathlib import Path

from racetrack.runtime import FlattenedDeepSeekModel
from racetrack.runtime.modeling import model_config

MODEL_NAME = "dsv3_2"
PARTITION_HASH = "a1f6d7e2"
PARTITION_NOTES = (
    "Fuses q/kv RMSNorm and RoPE at the MLA/indexer boundary, matching the "
    "shape of the monolithic attention path introduced in vLLM PR 38595."
)


def build_model(**overrides) -> FlattenedDeepSeekModel:
    config = model_config(MODEL_NAME).for_benchmark(**overrides)
    return FlattenedDeepSeekModel(config, partition_root=Path(__file__).parent)
