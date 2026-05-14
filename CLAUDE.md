# Racetrack

Partition and kernel benchmark harness for flattened DeepSeek-style models.
The goal is to find the fastest model/partition/kernel combination for each
benchmark workload.

## Concepts

- **Baseline**: `partitions/<model>/model.py` -- self-contained model with
  pure-torch ops, no kernel dispatch. This is the thing to beat.
- **Partition**: `partitions/<model>/<hash>/model.py` -- a modified copy of
  the baseline that routes one or more ops through `KernelDispatcher`. The
  `<hash>` is the first 8 hex chars of SHA-256 of the model.py content.
- **Kernel**: `partitions/<model>/<hash>/kernels/<backend>/<op>.py` -- a
  backend-specific implementation of a dispatchable op. Each file exports
  `BACKEND_AVAILABLE: bool` and one or more op functions.
- **Benchmark**: `benchmarks/<name>/` -- a folder with `bench.py` (runner)
  and `results/<model>/<hardware>.md` (results per model + hardware config,
  e.g. `results/dsv3_2/8xh100.md`). Run with `python -m benchmarks.<name>.bench`.

## Creating a new partition

1. Copy the baseline: `cp partitions/dsv3_2/model.py new_model.py`

2. Identify ops to fuse or replace. Look for sequences of small ops in the
   forward pass that could be a single kernel (e.g., norm + activation,
   attention score + mask + softmax, residual + norm). Each dispatchable op
   needs a torch fallback function already defined in the file and a call
   through the dispatcher:
   ```python
   if self.dispatcher is None:
       result = my_fused_op(x, weight, eps=eps)
   else:
       result = self.dispatcher.call("my_fused_op", my_fused_op, x, weight, eps=eps)
   ```

3. Hash and create the partition directory:
   ```bash
   HASH=$(python3 -c "import hashlib; print(hashlib.sha256(open('new_model.py','rb').read()).hexdigest()[:8])")
   mkdir -p partitions/dsv3_2/$HASH/kernels
   mv new_model.py partitions/dsv3_2/$HASH/model.py
   ```

4. Update `build_model()` at the bottom to pass `partition_root`:
   ```python
   def build_model(**overrides) -> FlattenedDeepSeekModel:
       config = DSV3_2_CONFIG.for_benchmark(**overrides)
       return FlattenedDeepSeekModel(config, partition_root=Path(__file__).parent)
   ```

5. Add `PARTITION_HASH` and `PARTITION_NOTES` constants at the top explaining
   what this partition changes vs baseline.

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

## Benchmarking

```bash
# Run a specific benchmark
python -m benchmarks.gsm8k.bench

# Run with specific device/dtype
python -m benchmarks.gsm8k.bench --device cuda:0 --dtype bfloat16

# The old sweep runner still works too
python -m racetrack.bench --partition all --kernel-filter all --benchmark smoke
```

After a benchmark run:
- `benchmarks/<name>/results/<model>/<hardware>.md` has the winner, full
  leaderboard, baseline comparison, and hardware info. The hardware slug is
  auto-detected (e.g., `results/dsv3_2/8xh100.md`). These files are committed.
- `partitions/<model>/<hash>/kernels/best.json` is a runtime cache for per-op
  backend winners (gitignored, not committed -- hardware-specific).

## Iteration loop

The workflow for improving performance:

1. Run the benchmark, check `winner.json`
2. If baseline wins, the partition's kernel overhead exceeds its fusion benefit
3. Analyze where time is spent (try larger sequence lengths or model dims)
4. Either:
   - Add more fused ops to an existing partition (new kernel files)
   - Create a new partition with different fusion boundaries
   - Write better kernel implementations for existing ops
5. Re-run the benchmark
6. Commit `results/<model>/<hardware>.md` when a partition beats baseline

## Partition files are self-contained

Every `model.py` inlines all dependencies (ops, config, dispatcher, model
classes). No imports from `racetrack.runtime`. This means duplication across
partitions, which is intentional -- each partition is independently readable
and modifiable without cross-file coordination.

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
