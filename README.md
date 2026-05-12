# racetrack

`racetrack` is a small benchmark harness for trying DeepSeek-style model
partitions against swappable kernel backends. It is inspired by the vLLM
specialized model work in:

- DeepSeek V3.2 NVFP4 specialized path: https://github.com/vllm-project/vllm/pull/38595
- DeepSeek V4 model path: https://github.com/vllm-project/vllm/pull/40860

The repo intentionally uses flattened, synthetic torch models instead of real
checkpoint loaders. The goal is to benchmark partition shapes and kernel
boundaries quickly: MLA projection/norm/RoPE, sparse-indexer-like staging,
hyper-compressed residual mixing, and routed MoE.

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
  dsv4/
    model.py
    b49c2a81/
      model.py
      kernels/{triton,cutedsl,helion}/fused_rope.py
  ds/
    model.py
    c7d9e510/
      model.py
      kernels/{triton,cutedsl,helion}/fused_rope.py
racetrack/
  bench.py                    # benchmark runner
  runtime/                    # dispatch, flattened model, torch/triton kernels
```

Each model folder has a baseline `model.py` that uses only torch operations.
Each partition folder is named like a content hash and has its own `model.py`
plus partition-local kernel entrypoints. The current partitions wrap fused
RMSNorm/RoPE and V4 `hc_head` calls through the kernel dispatcher.

## Kernel Selection

Set `RACETRACK_KERNEL_BACKEND` or pass `--kernel-filter`:

- `torch`: vanilla torch fallback
- `triton`: uses a native Triton RoPE kernel on CUDA, falling back to torch where needed
- `cutedsl` or `cutedl`: CUTEDSL adapter; requires the CUTLASS DSL package
- `helion`: Helion adapter; requires the Helion package
- `best`: runtime mixed-kernel mode; each kernel callsite is timed across
  Triton, CUTEDSL, and Helion, then the winner is cached for that callsite
- `all`: runner-only option that reports `triton`, `cutedsl`, `helion`, and a
  final `best` row. The `best` row is the fastest end-to-end measured
  candidate among the concrete backends and the runtime mixed-kernel plan.

For `best`, the status column reports what won. `pure=helion` means the pure
Helion run was the fastest end-to-end candidate. If the mixed-kernel plan wins,
the status includes the selected callsite plan, for example
`mixed=fused_norm_rope=helion;hc_head=cutedsl`. If the mixed planner selects
the same backend for every callsite, the end-to-end report uses the pure backend
row for clarity.

CUTEDSL and Helion are optional dependencies. Explicitly selecting a backend
whose package is unavailable is an error. The runner reports unavailable
backends as `missing`; it does not silently substitute torch.

## Quick Start

```bash
python -m pip install -e .
python -m pip install helion nvidia-cutlass-dsl
python -m racetrack.bench --model dsv3_2 --partition all --kernel-filter all
```

Run every initial model and backend on one GPU:

```bash
python -m racetrack.bench \
  --model all \
  --partition all \
  --kernel-filter all \
  --benchmark smoke \
  --device cuda:0 \
  --dtype bfloat16
```

Run the smoke sweep across an 8xH100 node:

```bash
python -m racetrack.bench \
  --model all \
  --partition all \
  --kernel-filter all \
  --benchmark smoke \
  --devices 0,1,2,3,4,5,6,7 \
  --dtype bfloat16
```

Run the realistic-shape distributed suite across all 8 H100s:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m racetrack.realistic_bench \
  --model all \
  --backend all \
  --tokens 1 \
  --layers realistic \
  --warmup 1 \
  --repeat 1 \
  --dtype bf16 \
  --json results/realistic_all_8h100.json
```

`racetrack.realistic_bench` uses DeepSeek-scale tensor dimensions:
hidden size 7168, 61 layers, 128 attention heads, 256 routed experts, top-k 8,
and model-specific MLA/V4 hyper-compression dimensions. It is still a synthetic
shape benchmark: it does not load Hugging Face checkpoints or allocate a full
61-layer checkpoint. Each rank owns its sharded heads and its 32-expert MoE
shard, then reuses one synthetic layer across the requested layer count so the
benchmark can exercise realistic matrix sizes, routing, kernel dispatch, and
NCCL all-reduces without checkpoint-scale memory.

Example 8xH100 realistic-shape output:

```text
model   backend  status        gpus  layers  tokens  mean_ms  tok*layer/s  diff       peak_gib  ok
dsv3_2  triton   native        8     61      1       154.847  393.9        0.000e+00  2.78      yes
dsv3_2  cutedsl  native        8     61      1       153.249  398.0        0.000e+00  2.78      yes
dsv3_2  helion   native        8     61      1       154.785  394.1        0.000e+00  2.78      yes
dsv3_2  best     pure=cutedsl  8     61      1       153.249  398.0        0.000e+00  2.78      yes
dsv4    triton   native        8     61      1       190.043  321.0        0.000e+00  2.77      yes
dsv4    cutedsl  native        8     61      1       189.871  321.3        0.000e+00  2.77      yes
dsv4    helion   native        8     61      1       189.440  322.0        0.000e+00  2.77      yes
dsv4    best     pure=helion   8     61      1       189.440  322.0        0.000e+00  2.77      yes
ds      triton   native        8     61      1       165.935  367.6        0.000e+00  2.78      yes
ds      cutedsl  native        8     61      1       166.282  366.8        0.000e+00  2.78      yes
ds      helion   native        8     61      1       166.467  366.4        0.000e+00  2.78      yes
ds      best     mixed=triton  8     61      1       165.764  368.0        0.000e+00  2.78      yes
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

1. Add a new directory under `partitions/<model>/<hash>/`.
2. Put the partition model in `model.py`.
3. Put backend kernels under `kernels/triton`, `kernels/cutedsl`, and
   `kernels/helion`.
4. Keep the public builder as `build_model(**overrides)`.
5. Run:

```bash
python -m racetrack.bench --model <model> --partition all --kernel-filter all
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
python -m racetrack.bench --model all --partition all --kernel-filter all --device cpu --dtype float32
```

The flattened models are deliberately small by default so kernel dispatch,
correctness checks, and sweeps iterate quickly. Override dimensions through the
`build_model(**overrides)` API when adding custom experiments.
