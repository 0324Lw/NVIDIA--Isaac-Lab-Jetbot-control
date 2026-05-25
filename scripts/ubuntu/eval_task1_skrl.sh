#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/ubuntu/eval_task1_skrl.sh /path/to/checkpoint_or_final_checkpoint_dir"
  echo ""
  echo "Examples:"
  echo "  bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run>/final_checkpoint"
  echo "  bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run>/final_checkpoint/diff_drive_task1_model.pt"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"

echo "============================================================"
echo "Diff-Drive UGV / Jetbot Task1 TRUE skrl PPO model evaluation"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CKPT}"
echo "PYTHON=$(which python)"
echo "============================================================"

python - <<'PY'
import sys
print("[CHECK] Python:", sys.executable)
import torch
print("[CHECK] torch:", torch.__version__)
print("[CHECK] cuda:", torch.cuda.is_available())
import isaaclab
print("[CHECK] isaaclab: ok")
import skrl
print("[CHECK] skrl:", getattr(skrl, "__version__", "unknown"))
PY

python src/diff_drive_rl/tasks/task1/task1_model_test.py \
  --checkpoint "${CKPT}" \
  --num-envs 4 \
  --steps 200 \
  --print-interval 20 \
  --max-episode-length-s 12.0 \
  --num-waypoints 3 \
  --headless \
  --device cuda:0
