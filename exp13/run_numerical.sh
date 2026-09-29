#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
base=exp13/results
if [[ ${1:-} == --smoke ]]; then base=exp13/results_smoke; fi
export MPLCONFIGDIR="$PWD/$base/stage1_numerical/matplotlib"
mkdir -p "$MPLCONFIGDIR"
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n curve python -B exp13/numerical.py "$@"
