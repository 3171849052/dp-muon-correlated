#!/usr/bin/env bash
# Launch the manifest-defined Experiment 1 runs in three serial GPU workers.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT="$ROOT/exp1/run.sh"
MANIFEST="$ROOT/exp1/manifest.tsv"
DONE_DIR="$ROOT/exp1/logs/.done"
SESSION="exp1"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

run_worker() {
  local worker_gpu=$1
  local id algorithm lr_setting clip_norm gpu config

  [[ -f "$MANIFEST" ]] || { echo "manifest does not exist: $MANIFEST; run python exp1/generate_configs.py first" >&2; exit 1; }
  mkdir -p "$DONE_DIR"
  cd "$ROOT"
  while IFS=$'\t' read -r id algorithm lr_setting clip_norm gpu config; do
    [[ "$id" == "id" || "$gpu" != "$worker_gpu" ]] && continue
    if [[ -e "$DONE_DIR/$id.done" ]]; then
      echo "[$(date -Is)] skipping completed $id"
      continue
    fi
    echo "[$(date -Is)] GPU $gpu: starting $id"
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_cifar10.py --config "$ROOT/$config"
    touch "$DONE_DIR/$id.done"
    echo "[$(date -Is)] GPU $gpu: completed $id"
  done < "$MANIFEST"
}

if [[ ${1:-} == "--worker" ]]; then
  [[ $# -eq 2 && $2 =~ ^[0-2]$ ]] || { echo "usage: $0 --worker GPU" >&2; exit 2; }
  run_worker "$2"
  exit 0
fi

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required to launch Exp1" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
[[ -f "$MANIFEST" ]] || { echo "manifest does not exist: $MANIFEST; run python exp1/generate_configs.py first" >&2; exit 1; }

printf -v worker_command 'exec bash %q --worker %q' "$SCRIPT" 0
tmux new-session -d -s "$SESSION" -n gpu0 "$worker_command"
printf -v worker_command 'exec bash %q --worker %q' "$SCRIPT" 1
tmux new-window -t "$SESSION" -n gpu1 "$worker_command"
printf -v worker_command 'exec bash %q --worker %q' "$SCRIPT" 2
tmux new-window -t "$SESSION" -n gpu2 "$worker_command"

echo "tmux session: $SESSION"
echo "workers: gpu0, gpu1, gpu2 (manifest-driven)"
echo "attach: tmux attach -t $SESSION"
