from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    tokens: int
    warmup: int = 5
    repeat: int = 20


_CASES: dict[str, BenchmarkCase] = {
    "smoke": BenchmarkCase("smoke", tokens=16, warmup=1, repeat=2),
    "decode_128": BenchmarkCase("decode_128", tokens=128),
    "prefill_512": BenchmarkCase("prefill_512", tokens=512),
    "prefill_2048": BenchmarkCase("prefill_2048", tokens=2048, warmup=3, repeat=10),
}


def get_cases(filter_name: str = "smoke") -> list[BenchmarkCase]:
    if filter_name == "all":
        return list(_CASES.values())
    if filter_name not in _CASES:
        known = ", ".join(sorted(_CASES))
        raise KeyError(f"Unknown benchmark case {filter_name!r}. Known: {known}, all")
    return [_CASES[filter_name]]
