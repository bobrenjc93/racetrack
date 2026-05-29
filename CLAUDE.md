# Racetrack

Partition and kernel benchmark harness for flattened DeepSeek-style models.
The goal is to find the fastest model/partition/kernel combination for each
benchmark workload.

## Concepts

- **Baseline**: `partitions/<model>/model.py` -- self-contained model with
  pure-torch ops, no kernel dispatch. This is the thing to beat.
- **Partition**: `partitions/<model>/<hash>/spec.py` -- a declarative spec
  defining which ops to fuse, how to patch the model pre-trace, and which
  FX graph patterns to match. The `<hash>` is the first 8 hex chars of
  SHA-256 of the spec content.
- **Kernel**: `partitions/<model>/<hash>/kernels/<backend>/<op>.py` -- a
  backend-specific implementation of a dispatchable op. Each file exports
  `BACKEND_AVAILABLE: bool` and one or more op functions.
- **Benchmark**: `benchmarks/<name>/` -- a folder with `bench.py` (runner)
  and `results/<model>/<hardware>.md` (results per model + hardware config,
  e.g. `results/dsv3_2/8xh100.md`). Run with `python -m benchmarks.<name>.bench`.

## Partition Architecture

Each partition is defined by `spec.py` + `kernels/`, driven by a
`torch.compile` custom backend:

```
partitions/<model>/<hash>/
  spec.py                    -- partition spec (FUSED_OPS, GRAPH_NODES)
  kernels/
    triton/<op>.py           -- Triton kernel implementations
    helion/<op>.py            -- Helion kernel implementations
    cutedsl/<op>.py           -- CuteDSL kernel implementations
```

The system uses three kinds of fused ops:

| Kind | When Applied | Examples |
|------|-------------|----------|
| `fx_pattern` | FX graph pattern matching after Dynamo traces | swiglu, residual_norm, act_quant |
| `pre_trace` | Before Dynamo traces (control flow changes) | indexer shortcircuit, single-token MoE |
| `module_patch` | Before Dynamo traces (module forward replacement) | mlp_gate_up_proj, attn_norm_qkv |

Pipeline:
1. Load spec → `PartitionSpec`
2. `apply_pre_trace_patches(model, spec, dispatcher)` — patches modules
3. `torch.compile(model, backend=RacetrackBackend(spec))` — traces + FX rewrites
4. Execute compiled model (optionally via CUDA graph)

## Creating a new partition

Use `scripts/gen_partition.py` to generate a partition from fusion recipes:

```bash
python scripts/gen_partition.py \
    --fuse rms_norm_q,rms_norm_kv,rope_kpe \
    --fuse res_add_attn,ffn_norm

python scripts/gen_partition.py --list-recipes  # show available recipes
python scripts/gen_partition.py --list-nodes    # show graph node IDs
```

This creates `partitions/<model>/<hash>/spec.py` plus one stub per backend at
`kernels/<backend>/ops.py`. Implement the kernels there; you may later split ops
across multiple files since the dispatcher scans all non-`_` `.py` files under
`kernels/<backend>/`.

## Writing kernels

Each kernel file lives at `kernels/<backend>/<name>.py` (e.g.,
`kernels/triton/attention.py`). It must export:

- `BACKEND_AVAILABLE: bool` -- whether the backend can run (check imports)
- One or more op functions matching the dispatcher op names

Op function signature must match the fallback, plus a `fallback` kwarg:
```python
def my_fused_op(x, weight, *, eps, fallback):
    del fallback  # not needed, we have our own implementation
    # ... kernel implementation ...
```

The dispatcher scans ALL `.py` files (excluding `_`-prefixed) under
`kernels/<backend>/` for each op, so kernels can be split across files.
Backends: `triton`, `helion`, `cutedsl`.

## Key modules

- `racetrack/partition_spec.py` -- PartitionSpec data model, loader, discovery
- `racetrack/pre_trace.py` -- Pre-trace model patchers (registry pattern)
- `racetrack/fx_patterns.py` -- FX graph pattern matchers
- `racetrack/compile_backend.py` -- Unified torch.compile backend
- `racetrack/runtime/dispatch.py` -- KernelDispatcher (backend selection)

## Benchmarking

```bash
# Run the real-weight GSM8K benchmark
torchrun --standalone --nproc-per-node=8 -m benchmarks.gsm8k.real_bench \
    --hf-direct --partition-model dsv3_2 --backend all

# Run the synthetic sweep benchmark
python -m racetrack.bench --partition all --kernel-filter all --benchmark smoke
```

After a benchmark run:
- `benchmarks/<name>/results/<model>/<hardware>.md` has the winner, full
  leaderboard, baseline comparison, and hardware info.
- `partitions/<model>/<hash>/kernels/best.json` is a runtime cache for per-op
  backend winners (gitignored, not committed -- hardware-specific).

## Environment

```bash
pip install -e .              # install racetrack
pip install triton            # for triton kernels
pip install helion            # for helion kernels
```

Set `RACETRACK_KERNEL_BACKEND` to control dispatch:
- `torch` -- pure PyTorch fallbacks only
- `triton` / `helion` / `cutedsl` -- force a specific backend
- `best` -- benchmark all backends on first call, cache winner to `best.json`
- `all` -- (bench.py only) sweep all backends
