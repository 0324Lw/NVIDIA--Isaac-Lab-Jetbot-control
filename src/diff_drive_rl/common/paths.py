from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
LOGS_ROOT = PROJECT_ROOT / "logs"
ASSETS_ROOT = PROJECT_ROOT / "assets"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
