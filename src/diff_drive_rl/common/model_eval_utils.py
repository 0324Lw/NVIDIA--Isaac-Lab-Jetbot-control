from __future__ import annotations

from pathlib import Path
import torch


def torch_load_checkpoint(path: str | Path, device: str = "cpu"):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def resolve_checkpoint(path: str | Path, preferred_name: str = "") -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        if preferred_name:
            candidate = p / preferred_name
            if candidate.exists():
                return candidate
        for candidate in sorted(p.glob("*.pt")):
            if candidate.name.endswith("_preprocessor.pt"):
                continue
            if candidate.name in {"train_metadata.pt"}:
                continue
            return candidate
    return p
