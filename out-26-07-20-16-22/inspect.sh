#!/usr/bin/env bash
# Inspect a torch-dag-gfn run directory: ground truth, posterior samples, data, config, metrics.
#
#   ./inspect.sh            # inspect the directory this script lives in
#   ./inspect.sh ../out-XX  # inspect another run directory
#
# Needs numpy only (the `torch-dag-gfn` conda env works).
set -euo pipefail

RUN_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

python - "$RUN_DIR" <<'PY'
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
print(f"run directory: {run}\n")


def section(title):
    print(f"=== {title} " + "=" * max(0, 60 - len(title)))


def matrix(array, **kwargs):
    return textwrap.indent(np.array2string(array, **kwargs), "  ")


section("config (arguments.json)")
args = json.loads((run / "arguments.json").read_text())
for key, value in args.items():
    print(f"  {key:<22} {value}")
print()

section("ground truth (ground_truth.npy)")
truth = np.load(run / "ground_truth.npy")
print(f"  shape {truth.shape}  dtype {truth.dtype}   # A[i, j] == 1 means edge i -> j")
print(matrix(truth.astype(int)))
edges = [(int(i), int(j)) for i, j in zip(*truth.nonzero())]
print(f"  {len(edges)} edges: " + ", ".join(f"{i} -> {j}" for i, j in edges))
print()

section("posterior samples (posterior.npy)")
posterior = np.load(run / "posterior.npy")
print(f"  shape {posterior.shape}  dtype {posterior.dtype}   # (num_samples, d, d)")
marginals = posterior.mean(axis=0)
print("  marginal edge probabilities P(i -> j):")
print(matrix(marginals, precision=3, suppress_small=True))
print(f"  mean edge count per sampled graph: {posterior.sum(axis=(1, 2)).mean():.3f}")
print("  top edges by marginal probability (* = in ground truth):")
order = np.dstack(np.unravel_index(np.argsort(marginals, axis=None)[::-1], marginals.shape))[0]
for i, j in order[:10]:
    mark = "*" if truth[i, j] else " "
    print(f"    {mark} {i} -> {j}   {marginals[i, j]:.3f}")
print()

section("data (data.npz)")
with np.load(run / "data.npz") as data:
    for key in data.files:
        array = data[key]
        print(f"  {key:<14} shape {str(array.shape):<10} dtype {array.dtype}")
    print(f"  variable kinds: {data['kinds'].tolist()}")
    print(f"  cardinalities:  {data['cardinalities'].tolist()}   # 0 for continuous variables")
print()

section("metrics (results.json)")
for key, value in json.loads((run / "results.json").read_text()).items():
    print(f"  {key:<18} {value}")
PY
