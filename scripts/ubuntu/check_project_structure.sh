#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "============================================================"
echo "Diff-Drive UGV project structure check"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "============================================================"

required=(
  "README.md"
  "CHANGELOG.md"
  "CONTRIBUTING.md"
  "LICENSE"
  "pyproject.toml"
  ".gitignore"

  "assets/README.md"
  "assets/gifs/README.md"
  "assets/motions/README.md"
  "assets/usd/README.md"

  "configs/.gitkeep"
  "configs/local_paths.example.yaml"
  "configs/platform_ubuntu_laptop.yaml"
  "configs/platform_windows_3090.yaml"
  "configs/task1_multi_waypoint_navigation.yaml"
  "configs/task2_obstacle_navigation.yaml"
  "configs/task3_sim2real_parking.yaml"
  "configs/task4_multi_ugv_formation_escort.yaml"

  "docs/project_overview.md"
  "docs/results_and_checkpoints.md"
  "docs/task1_design.md"
  "docs/task2_design.md"
  "docs/task3_design.md"
  "docs/task4_design.md"
  "docs/troubleshooting.md"
  "docs/ubuntu_training.md"
  "docs/windows_path_config.md"
  "docs/windows_training.md"

  "scripts/ubuntu/eval_task1_skrl.sh"
  "scripts/ubuntu/eval_task2_skrl.sh"
  "scripts/ubuntu/eval_task3_skrl.sh"
  "scripts/ubuntu/eval_task4_skrl.sh"
  "scripts/ubuntu/test_task1_env.sh"
  "scripts/ubuntu/test_task2_env.sh"
  "scripts/ubuntu/test_task3_env.sh"
  "scripts/ubuntu/test_task4_env.sh"
  "scripts/ubuntu/train_task1_skrl_smoke.sh"
  "scripts/ubuntu/train_task2_skrl_smoke.sh"
  "scripts/ubuntu/train_task3_skrl_smoke.sh"
  "scripts/ubuntu/train_task4_skrl_smoke.sh"

  "scripts/windows/check_task1_windows_ready.ps1"
  "scripts/windows/check_task2_windows_ready.ps1"
  "scripts/windows/check_task3_windows_ready.ps1"
  "scripts/windows/check_task4_windows_ready.ps1"
  "scripts/windows/eval_task1_skrl_3090.ps1"
  "scripts/windows/eval_task2_skrl_3090.ps1"
  "scripts/windows/eval_task3_skrl_3090.ps1"
  "scripts/windows/eval_task4_skrl_3090.ps1"
  "scripts/windows/train_task1_skrl_3090.ps1"
  "scripts/windows/train_task1_skrl_smoke_3090.ps1"
  "scripts/windows/train_task2_skrl_3090.ps1"
  "scripts/windows/train_task2_skrl_smoke_3090.ps1"
  "scripts/windows/train_task3_skrl_3090.ps1"
  "scripts/windows/train_task3_skrl_smoke_3090.ps1"
  "scripts/windows/train_task4_skrl_3090.ps1"
  "scripts/windows/train_task4_skrl_smoke_3090.ps1"

  "src/diff_drive_rl/common/__init__.py"
  "src/diff_drive_rl/common/eval_curriculum_utils.py"
  "src/diff_drive_rl/common/diff_drive_skrl_models.py"
  "src/diff_drive_rl/common/diff_drive_skrl_wrappers.py"
  "src/diff_drive_rl/common/info_utils.py"
  "src/diff_drive_rl/common/model_eval_utils.py"
  "src/diff_drive_rl/common/paths.py"
  "src/diff_drive_rl/common/progress.py"
  "src/diff_drive_rl/common/running_mean_std.py"
  "src/diff_drive_rl/common/skrl_models.py"
  "src/diff_drive_rl/common/vec_wrappers.py"

  "src/diff_drive_rl/data/__init__.py"
  "src/diff_drive_rl/data/README.md"

  "src/diff_drive_rl/tasks/task1/task1_config.py"
  "src/diff_drive_rl/tasks/task1/task1_env.py"
  "src/diff_drive_rl/tasks/task1/task1_train.py"
  "src/diff_drive_rl/tasks/task1/task1_model_test.py"
  "src/diff_drive_rl/tasks/task2/task2_config.py"
  "src/diff_drive_rl/tasks/task2/task2_env.py"
  "src/diff_drive_rl/tasks/task2/task2_train.py"
  "src/diff_drive_rl/tasks/task2/task2_model_test.py"
  "src/diff_drive_rl/tasks/task3/task3_config.py"
  "src/diff_drive_rl/tasks/task3/task3_env.py"
  "src/diff_drive_rl/tasks/task3/task3_train.py"
  "src/diff_drive_rl/tasks/task3/task3_model_test.py"
  "src/diff_drive_rl/tasks/task4/task4_config.py"
  "src/diff_drive_rl/tasks/task4/task4_env.py"
  "src/diff_drive_rl/tasks/task4/task4_train.py"
  "src/diff_drive_rl/tasks/task4/task4_model_test.py"

  "tests/task1/task1_env_test.py"
  "tests/task2/task2_env_test.py"
  "tests/task3/task3_env_test.py"
  "tests/task4/task4_env_test.py"
)

missing=0
for p in "${required[@]}"; do
  if [ -e "$p" ]; then
    echo "[OK] $p"
  else
    echo "[MISSING] $p"
    missing=$((missing + 1))
  fi
done

if [ "$missing" -ne 0 ]; then
  echo "[FAIL] missing_count=${missing}"
  exit 1
fi

echo "[PASS] project structure check passed"
