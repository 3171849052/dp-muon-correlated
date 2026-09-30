#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
config_snapshot=$(mktemp)
cp exp13/config.yaml "$config_snapshot"
export EXP13_CONFIG="$config_snapshot"
trap 'rm -f "$config_snapshot"' EXIT
base=exp13/results
if [[ ${1:-} == --smoke ]]; then base=exp13/results_smoke; fi
root="$base/stage2_training"
export MPLCONFIGDIR="$PWD/$root/matplotlib"
export TMPDIR="$PWD/$root/tmp"
mkdir -p "$root/worker_logs" "$MPLCONFIGDIR" "$TMPDIR"
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n curve python -B exp13/validate_strategies.py "$@"
pids=()
for gpu in 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n curve python -B exp13/worker.py --gpu "$gpu" "$@" > "$root/worker_logs/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else status=1; fi
done
if (( status != 0 )); then exit "$status"; fi
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n curve python -B exp13/aggregate.py "$@"
