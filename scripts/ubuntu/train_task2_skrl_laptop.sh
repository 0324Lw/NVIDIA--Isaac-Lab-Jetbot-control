#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

python src/diff_drive_rl/tasks/task2/task2_train.py \
  --num-envs 4096 \
  --total-env-steps 250000000 \
  --rollouts 64 \
  --learning-epochs 4 \
  --mini-batches 8 \
  --lr 2.5e-4 \
  --min-lr 5e-5 \
  --max-lr 5e-4 \
  --gamma 0.995 \
  --gae-lambda 0.95 \
  --clip-range 0.20 \
  --target-kl 0.020 \
  --hard-kl-stop 0.120 \
  --entropy-coef 0.004 \
  --value-coef 1.0 \
  --grad-clip 0.5 \
  --init-log-std -0.70 \
  --max-episode-length-s 80.0 \
  --summary-interval 1 \
  --save-freq-env-steps 10000000 \
  --headless \
  --device cuda:0
