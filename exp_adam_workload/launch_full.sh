#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p exp_adam_workload/results/worker_logs
# Preparation and clean replay finish before workers consume shared artifacts.
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n curve python exp_adam_workload/run.py
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n curve \
    python exp_adam_workload/run_full.py --worker-index "$gpu" --num-workers 4 \
    >"exp_adam_workload/results/worker_logs/worker${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
