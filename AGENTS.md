# Agent Guidelines

## After creating or modifying a partition

Always run `python scripts/gen_graphs.py` and commit the resulting `graph.png`
files. Every partition directory must contain a `graph.png` showing its fused
op flow. The script reads the `FUSED_OP_GRAPH` dict from each partition's
`model.py`, so make sure that constant is defined before running the script.
