# Real Arithmetic Benchmark: 8xh100

**Model**: deepseek-ai/DeepSeek-V3.2
**Partition model**: dsv3_2
**GPU**: NVIDIA H100
**PyTorch**: 2.13.0a0+git8fc3c90
**Date**: 2026-05-25T22:45:09.134123+00:00
**Decode tokens**: 256 (4 prompts x 64 tokens)

## Winner

**3336cdbd/triton** (2.063x vs baseline)
Aggregate: 34676.8ms

## Leaderboard

| # | partition | backend | total (ms) | vs baseline |
|---|---|---|---|---|
| 1 | 3336cdbd | triton | 34676.8 | 2.063x |
| 2 | 3336cdbd | cutedsl | 35262.9 | 2.029x |
| 3 | baseline | torch.compile | 36741.1 | 1.947x |
| 4 | baseline | torch | 71547.8 | 1.000x |

