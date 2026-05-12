from __future__ import annotations

from racetrack.realistic_bench import realistic_shape


def test_realistic_dsv3_2_shape() -> None:
    shape = realistic_shape("dsv3_2")

    assert shape.hidden_size == 7168
    assert shape.num_layers == 61
    assert shape.num_attention_heads == 128
    assert shape.n_routed_experts == 256
    assert shape.num_experts_per_tok == 8
    assert shape.q_lora_rank == 1536
    assert shape.kv_lora_rank == 512


def test_realistic_dsv4_shape() -> None:
    shape = realistic_shape("dsv4")

    assert shape.hidden_size == 7168
    assert shape.num_layers == 61
    assert shape.num_attention_heads == 128
    assert shape.n_routed_experts == 256
    assert shape.num_experts_per_tok == 8
    assert shape.q_lora_rank == 1536
    assert shape.kv_lora_rank == 128
    assert shape.hc_mult == 2
