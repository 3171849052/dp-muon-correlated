#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The project uses a ``src`` layout and may be run without an editable install.
# Keep the source tree importable for both the setup queries below and the
# command launched inside tmux.
# cifar10_nonamplified cifar10_dpsgd_momentum
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
DEFAULT_CONFIG="$ROOT/config/cifar10_dpsgd_momentum.yaml"

if [[ $# -eq 0 ]]; then
  CONFIG=$DEFAULT_CONFIG
elif [[ $# -eq 2 && $1 == "--config" ]]; then
  CONFIG=$2
elif [[ $# -eq 1 && $1 != "--config" ]]; then
  # Preserve the original positional form for existing launch commands.
  CONFIG=$1
else
  echo "usage: $0 [--config CONFIG]" >&2
  echo "       $0 [CONFIG]" >&2
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "config file does not exist: $CONFIG" >&2
  exit 1
fi

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
