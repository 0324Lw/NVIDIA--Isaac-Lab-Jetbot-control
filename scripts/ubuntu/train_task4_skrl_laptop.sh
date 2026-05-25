#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

NUM_ENVS="${1:-128}"
TOTAL_AGENT_STEPS="${2:-50000000}"
DEVICE="${3:-cuda:0}"

echo "============================================================"
echo "Diff-Drive UGV / Jetbot Task4 TRUE skrl PPO laptop training"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "NUM_ENVS=${NUM_ENVS} physical envs"
echo "TOTAL_AGENT_STEPS=${TOTAL_AGENT_STEPS}"
echo "DEVICE=${DEVICE}"
echo "NOTE: skrl agent envs = physical envs * 4"
echo "PYTHON=$(which python)"
echo "============================================================"

python src/diff_drive_rl/tasks/task4/task4_train.py \
  --num-envs "${NUM_ENVS}" \
  --total-agent-steps "${TOTAL_AGENT_STEPS}" \
  --save-freq-agent-steps 5000000 \
  --rollouts 64 \
  --learning-epochs 5 \
  --mini-batches 8 \
  --lr 8.0e-5 \
  --test-device "${DEVICE}" \
  --headless
