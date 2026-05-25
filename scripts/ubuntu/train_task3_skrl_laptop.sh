#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

python src/diff_drive_rl/tasks/task3/task3_train.py \
  --num-envs 1024 \
  --total-env-steps 300000000 \
  --rollouts 64 \
  --learning-epochs 5 \
  --mini-batches 8 \
  --lr 1.0e-4 \
  --min-lr 2.0e-5 \
  --max-lr 1.5e-4 \
  --gamma 0.995 \
  --gae-lambda 0.95 \
  --clip-range 0.18 \
  --target-kl 0.010 \
  --hard-kl-stop 0.120 \
  --entropy-coef 0.002 \
  --value-coef 1.0 \
  --grad-clip 0.5 \
  --init-log-std -1.0 \
  --min-log-std -3.0 \
  --max-log-std 0.5 \
  --max-episode-length-s 40.0 \
  --max-wheel-speed 14.0 \
  --summary-interval 1 \
  --save-freq-env-steps 10000000 \
  --headless \
  --device cuda:0
