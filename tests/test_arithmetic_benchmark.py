from __future__ import annotations

from benchmarks.arithmetic.bench import CASES, _validate_cases, run


def test_arithmetic_cases_are_identity_equations() -> None:
    _validate_cases(CASES)
    assert CASES[0].prompt == "1 * 213813290183291 ="
    assert CASES[0].expected == "213813290183291"


def test_arithmetic_cpu_baseline_torch() -> None:
    results = run(
        device_str="cpu",
        dtype_str="float32",
        warmup=0,
        repeat=1,
        partition_filter="baseline",
        kernel_filter="torch",
        check=True,
    )
    assert len(results) == len(CASES)
    assert all(result.ok for result in results)
    assert {result.backend for result in results} == {"torch"}
    assert [result.tokens for result in results] == [case.tokens for case in CASES]
