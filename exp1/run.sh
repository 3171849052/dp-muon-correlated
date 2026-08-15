#!/usr/bin/env bash
# Launch the nine Experiment 1 BandInvMF CIFAR-10 runs in three GPU workers.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT="$ROOT/exp1/run.sh"
CONFIG_DIR="$ROOT/exp1/config"
DONE_DIR="$ROOT/exp1/logs/.done"
SESSION="exp1"

# The project uses a ``src`` layout and may be run without an editable install.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

run_worker() {
  local gpu=$1
  local config
  local -a configs

  case "$gpu" in
    0) configs=(lr0.1_clip1 lr0.1_clip5 lr0.1_clip10) ;;
    1) configs=(lr0.5_clip1 lr0.5_clip5 lr0.5_clip10) ;;
    2) configs=(lr1.0_clip1 lr1.0_clip5 lr1.0_clip10) ;;
    *) echo "unknown Exp1 GPU worker: $gpu" >&2; exit 2 ;;
  esac

  mkdir -p "$DONE_DIR"
  cd "$ROOT"
  for config in "${configs[@]}"; do
    if [[ -e "$DONE_DIR/$config.done" ]]; then
      echo "[$(date -Is)] skipping completed $config"
      continue
    fi

    echo "[$(date -Is)] GPU $gpu: starting $config"
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_cifar10.py --config "$CONFIG_DIR/$config.yaml"
    touch "$DONE_DIR/$config.done"
    echo "[$(date -Is)] GPU $gpu: completed $config"
  done
}

if [[ ${1:-} == "--worker" ]]; then
  [[ $# -eq 2 ]] || { echo "usage: $0 --worker GPU" >&2; exit 2; }
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

printf -v worker_command 'exec bash %q --worker %q' "$SCRIPT" 0
tmux new-session -d -s "$SESSION" -n gpu0 "$worker_command"
printf -v worker_command 'exec bash %q --worker %q' "$SCRIPT" 1
tmux new-window -t "$SESSION" -n gpu1 "$worker_command"
printf -v worker_command 'exec bash %q --worker %q' "$SCRIPT" 2
tmux new-window -t "$SESSION" -n gpu2 "$worker_command"

echo "tmux session: $SESSION"
echo "workers: gpu0 (lr=0.1), gpu1 (lr=0.5), gpu2 (lr=1.0)"
echo "attach: tmux attach -t $SESSION"
