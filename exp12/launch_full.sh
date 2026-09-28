#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="$PWD/exp12/results/matplotlib"
export TMPDIR="$PWD/exp12/results/tmp"
mkdir -p exp12/results/worker_logs "$MPLCONFIGDIR" "$TMPDIR"
exec > >(tee -a exp12/results/worker_logs/launcher.log) 2>&1
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n curve python -B exp12/fit_strategies.py
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n curve python -B exp12/collect_trajectory.py
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n curve python -B exp12/replay.py
pids=()
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n curve python -B exp12/worker.py --gpu "$gpu" > "exp12/results/worker_logs/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else status=1; fi
done
if (( status != 0 )); then exit "$status"; fi
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n curve python -B exp12/aggregate.py
