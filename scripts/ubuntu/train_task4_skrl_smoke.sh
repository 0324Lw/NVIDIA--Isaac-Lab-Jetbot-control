#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

NUM_ENVS="${1:-512}"
TOTAL_AGENT_STEPS="${2:-5000}"
DEVICE="${3:-cuda:0}"

echo "============================================================"
echo "Diff-Drive UGV / Jetbot Task4 TRUE skrl PPO smoke training"
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
  --save-freq-agent-steps "${TOTAL_AGENT_STEPS}" \
  --rollouts 16 \
  --learning-epochs 2 \
  --mini-batches 4 \
  --lr 8.0e-5 \
  --fixed-stage 0 \
  --test-device "${DEVICE}" \
  --headless
