#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

echo "============================================================"
echo "Diff-Drive UGV / Jetbot Task4 Multi-UGV Formation Escort World Test"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON=$(which python)"
echo "============================================================"

python - <<'PY'
import sys
print("[CHECK] Python:", sys.executable)

try:
    import torch
    print("[CHECK] torch:", torch.__version__)
    print("[CHECK] cuda available:", torch.cuda.is_available())
except Exception as e:
    raise RuntimeError("Current Python cannot import torch. Please activate conda env: isaaclab") from e

try:
    import isaaclab
    print("[CHECK] isaaclab: ok")
except Exception as e:
    raise RuntimeError("Current Python cannot import isaaclab. Please activate IsaacLab conda env.") from e

try:
    import diff_drive_rl
    print("[CHECK] diff_drive_rl import: ok")
except Exception as e:
    raise RuntimeError("Cannot import diff_drive_rl. Check PYTHONPATH and project structure.") from e
PY

python tests/task4/task4_world_test.py \
  --num-envs 64 \
  --steps 2000 \
  --headless \
  --test-device cuda:0
