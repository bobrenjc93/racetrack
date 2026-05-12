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
  `best` row equal to the fastest measured concrete backend for that case

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
