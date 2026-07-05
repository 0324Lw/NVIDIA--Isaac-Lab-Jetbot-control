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

required_paths=(
  "src/diff_drive_rl"
  "src/diff_drive_rl/tasks/task1/task1_env.py"
  "src/diff_drive_rl/tasks/task2/task2_env.py"
  "src/diff_drive_rl/tasks/task3/task3_env.py"
  "src/diff_drive_rl/tasks/task4/task4_env.py"
  "tests/task1/task1_env_test.py"
  "tests/task2/task2_world_test.py"
  "tests/task3/task3_world_test.py"
  "tests/task4/task4_world_test.py"
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] missing path: ${required_path}" >&2
    exit 1
  fi
done
echo "[OK] project structure check passed"
