# Real Arithmetic Benchmark: 8xh100

**Model**: deepseek-ai/DeepSeek-V3.2
**Partition model**: dsv3_2_nvfp4
**GPU**: NVIDIA H100
**PyTorch**: 2.13.0a0+git8fc3c90
**Date**: 2026-05-25T23:46:14.758489+00:00
**Decode tokens**: 256 (4 prompts x 64 tokens)

## Winner

**f1bdaa6e/triton** (2.061x vs baseline)
Aggregate: 34683.2ms

## Leaderboard

| # | partition | backend | total (ms) | vs baseline |
|---|---|---|---|---|
| 1 | f1bdaa6e | triton | 34683.2 | 2.061x |
| 2 | f1bdaa6e | cutedsl | 35264.1 | 2.027x |
| 3 | baseline | torch.compile | 37894.8 | 1.886x |
| 4 | baseline | torch | 71467.6 | 1.000x |

