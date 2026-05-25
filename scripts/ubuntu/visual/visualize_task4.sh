#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/ubuntu/visual/visualize_task4.sh /path/to/checkpoint_or_final_checkpoint_dir [start_k] [slow_action_scale]"
  echo ""
  echo "Examples:"
  echo "  bash scripts/ubuntu/visual/visualize_task4.sh logs/task4/<run>/final_checkpoint 0.0 1.0"
  echo "  bash scripts/ubuntu/visual/visualize_task4.sh logs/task4/<run>/final_checkpoint 1.0 0.5"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"
START_K="${2:-0.0}"
SLOW_ACTION_SCALE="${3:-1.0}"

echo "============================================================"
echo "Diff-Drive UGV / Jetbot Task4 TRUE skrl PPO GUI visualization"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CKPT}"
echo "START_K=${START_K}"
echo "SLOW_ACTION_SCALE=${SLOW_ACTION_SCALE}"
echo "PYTHON=$(which python)"
echo "============================================================"

python src/diff_drive_rl/tasks/task4/task4_model_test.py \
  --checkpoint "${CKPT}" \
  --num-envs 4 \
  --steps 2000 \
  --start-k "${START_K}" \
  --print-interval 20 \
  --max-episode-length-s 35.0 \
  --slow-action-scale "${SLOW_ACTION_SCALE}" \
  --visualize \
  --device cuda:0
