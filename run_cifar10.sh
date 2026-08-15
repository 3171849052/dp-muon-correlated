#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIG" >&2
  exit 2
fi

CONFIG=$1
if [[ ! -f "$CONFIG" ]]; then
  echo "config file does not exist: $CONFIG" >&2
  exit 1
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG=$(realpath "$CONFIG")
LOG_DIR=$(cd "$ROOT" && python scripts/run_cifar10.py --config "$CONFIG" --print-log-dir)
GPU=$(cd "$ROOT" && python scripts/run_cifar10.py --config "$CONFIG" --print-gpu)
mkdir -p "$LOG_DIR"
SESSION="cifar10_$(basename "$CONFIG" .yaml)_$(printf '%s' "$CONFIG" | sha256sum | cut -c1-10)"
LOG="$LOG_DIR/${SESSION}.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

printf -v COMMAND \
  'cd %q && GPU=%q && CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_cifar10.py --config %q 2>&1 | tee -a %q' \
  "$ROOT" "$GPU" "$CONFIG" "$LOG"
tmux new-session -d -s "$SESSION" "$COMMAND"

echo "tmux session: $SESSION"
echo "physical GPU: $GPU"
echo "log: $LOG"
echo "attach: tmux attach -t $SESSION"
echo "tail: tail -f $LOG"
