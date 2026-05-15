# racetrack

`racetrack` is a small benchmark harness for trying DeepSeek-style model
partitions against swappable kernel backends. It is inspired by the vLLM
specialized model work in:

- DeepSeek V3.2 NVFP4 specialized path: https://github.com/vllm-project/vllm/pull/38595

The repo intentionally uses flattened, synthetic torch models instead of real
checkpoint loaders. The goal is to benchmark partition shapes and kernel
boundaries quickly: MLA projection/norm/RoPE, sparse-indexer-like staging, and
routed MoE.

## Layout

```text
benchmarks/
  suites.py
partitions/
  dsv3_2/
    model.py                  # vanilla torch baseline
    a1f6d7e2/
      model.py                # partition using backend-dispatched kernels
      kernels/
        triton/fused_rope.py
        cutedsl/fused_rope.py
        helion/fused_rope.py
racetrack/
  bench.py                    # benchmark runner
  runtime/                    # dispatch, flattened model, backend kernels
```

Each model folder has a baseline `model.py` that uses only torch operations.
Each partition folder is named like a content hash and has its own `model.py`
plus partition-local kernel entrypoints. The current partitions wrap fused
RMSNorm/RoPE calls through the kernel dispatcher.

## Kernel Selection

Set `RACETRACK_KERNEL_BACKEND` or pass `--kernel-filter`:

- `torch`: vanilla torch reference implementation
- `triton`: uses native Triton kernels for RMSNorm and RoPE on CUDA
- `cutedsl` or `cutedl`: uses native CuTe/CUTLASS DSL kernels for RMSNorm and RoPE on CUDA
- `helion`: uses Helion kernels for RMSNorm and RoPE, with Helion autotuning on first use
- `best`: runtime mixed-kernel mode; each kernel callsite is timed across
  implemented concrete backends, then the winner is cached for that callsite
- `all`: runner-only option that reports implemented concrete backends and a
  final `best` row. Today that means `triton`, `cutedsl`, `helion`, and `best`.

For `best`, the status column reports what won. `pure=helion` means the pure
Helion run was the fastest end-to-end candidate. If the mixed-kernel plan wins,
the status includes the selected callsite plan, for example
`mixed=fused_norm_rope=helion`. If the mixed planner selects the same backend
for every callsite, the end-to-end report uses the pure backend row for clarity.

CUTEDSL and Helion are optional dependencies. Explicitly selecting a backend
whose real kernel is unavailable is an error. The runner does not silently
substitute torch. Helion's default autotune effort is `quick`; set
`RACETRACK_HELION_AUTOTUNE_EFFORT=full` before running to search harder, or
`HELION_FORCE_AUTOTUNE=1` to ignore cached configs and re-run the search.

## Quick Start

```bash
python -m pip install -e .
python -m pip install helion nvidia-cutlass-dsl
python -m racetrack.bench --model dsv3_2 --partition all --kernel-filter all
```

Run every backend on one GPU:

```bash
python -m racetrack.bench \
  --model dsv3_2 \
  --partition all \
  --kernel-filter all \
  --benchmark smoke \
  --device cuda:0 \
  --dtype bfloat16
```

Run the smoke sweep across an 8xH100 node:

```bash
python -m racetrack.bench \
  --model dsv3_2 \
  --partition all \
  --kernel-filter all \
  --benchmark smoke \
  --devices 0,1,2,3,4,5,6,7 \
  --dtype bfloat16
```

Run the realistic-shape distributed suite across all 8 H100s:

```bash
RACETRACK_HELION_AUTOTUNE_EFFORT=quick \
HELION_FORCE_AUTOTUNE=1 \
torchrun --standalone --nproc-per-node=8 \
  -m racetrack.realistic_bench \
  --model dsv3_2 \
  --backend all \
  --tokens 1 \
  --layers realistic \
  --warmup 1 \
  --repeat 1 \
  --dtype bf16 \
  --rtol 1e-1 \
  --json results/realistic_all_8h100.json
```

`racetrack.realistic_bench` uses DeepSeek-scale tensor dimensions:
hidden size 7168, 61 layers, 128 attention heads, 256 routed experts, top-k 8,
and the V3.2 MLA dimensions. It is still a synthetic shape benchmark: it does
not load Hugging Face checkpoints or allocate a full 61-layer checkpoint. Each
rank owns its sharded heads and its 32-expert MoE shard, then reuses one
synthetic layer across the requested layer count so the benchmark can exercise
realistic matrix sizes, routing, kernel dispatch, and NCCL all-reduces without
checkpoint-scale memory.

The realistic runner checks both absolute and relative error:
`diff <= atol + rtol * max_abs(reference)`. Defaults are `--atol 0.5` and
`--rtol 1e-1`, which are intended for long bf16 recurrence checks.

## Real GSM8K leaderboard

Published GSM8K leaderboards should use the real-weight runner, not the
synthetic partition shape benchmark. The real runner loads the converted
DeepSeek-V3.2 model-parallel checkpoint, evaluates the full baseline model,
then patches compatible partition-local kernels into that same full model and
requires every patched row to reproduce the baseline generated completions.

```bash
torchrun --standalone --nproc-per-node=8 \
  -m benchmarks.gsm8k.real_bench \
  --ckpt-path checkpoints/dsv3_2-mp8 \
  --samples 50
```

The runner requires a Hugging Face token via `--hf-token`, `HF_TOKEN`, or
`hf_token=...` in `~/.env`. Rows are only emitted for partition kernels with a
real-model adapter; unsupported synthetic-only fusions are excluded until they
can be bound to checkpoint-backed `inference.model.Transformer` modules.
If the converted checkpoint is not available, pass `--hf-direct` to stream the
Hugging Face shards and slice the tensor-parallel weights on each rank.

Example 8xH100 realistic-shape output:

```text
model   backend  status        gpus  layers  tokens  mean_ms  tok*layer/s  diff       rel        peak_gib  ok
dsv3_2  triton   native        8     61      1       152.584  399.8        1.000e+00  1.562e-02  2.78      yes
dsv3_2  cutedsl  native        8     61      1       174.657  349.3        5.500e+00  8.594e-02  2.78      yes
dsv3_2  helion   native        8     61      1       151.924  401.5        1.000e+00  1.562e-02  2.98      yes
dsv3_2  best     mixed=helion  8     61      1       150.104  406.4        1.000e+00  1.562e-02  2.78      yes
```

Larger built-in benchmark cases are `decode_128`, `prefill_512`, and
`prefill_2048`.

## Output

The runner prints a compact table with model, partition, backend, backend
status, case, device, latency, throughput, and max absolute difference versus
the baseline. Add `--json results/smoke.json` to save structured results.

Example:

```text
model   partition  backend  status    case   device  mean_ms  tok/s   diff       ok
dsv3_2  baseline   torch    native    smoke  cuda:0  12.603   1269.5  0.000e+00  yes
dsv3_2  a1f6d7e2   triton   native    smoke  cuda:0  37.462   427.1   1.562e-02  yes
```

## Adding A Partition

1. Add a new directory under `partitions/dsv3_2/<hash>/`.
2. Put the partition model in `model.py`.
3. Put backend kernels under `kernels/triton`, `kernels/cutedsl`, and `kernels/helion`.
4. Keep the public builder as `build_model(**overrides)`.
5. Run:

```bash
python -m racetrack.bench --model dsv3_2 --partition all --kernel-filter all
```

A practical hash command for a new partition file is:

```bash
python - <<'PY'
import hashlib, pathlib
print(hashlib.sha256(pathlib.Path("model.py").read_bytes()).hexdigest()[:8])
PY
```

## Development Checks

```bash
python -m pytest
python -m racetrack.bench --model dsv3_2 --partition baseline --kernel-filter torch --device cpu --dtype float32
```

The flattened models are deliberately small by default so kernel dispatch,
correctness checks, and sweeps iterate quickly. Override dimensions through the
`build_model(**overrides)` API when adding custom experiments.
