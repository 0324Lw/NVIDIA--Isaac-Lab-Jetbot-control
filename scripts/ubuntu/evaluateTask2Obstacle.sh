#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
export PYTHONUNBUFFERED=1

CHECKPOINT_PATH="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(find logs/task2 -path '*/final_checkpoint' -type d 2>/dev/null | sort | tail -n 1)"
fi
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] checkpoint not found. Set CHECKPOINT=/path/to/checkpoint" >&2
  exit 1
fi
python -m diff_drive_rl.tasks.task2.task2_model_test --checkpoint "${CHECKPOINT_PATH}" "$@"
