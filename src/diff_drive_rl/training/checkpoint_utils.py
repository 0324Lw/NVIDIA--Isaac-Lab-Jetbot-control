from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def json_safe(value: Any) -> Any:
    """Convert common training objects to JSON-serializable values."""
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu()
        if data.numel() == 1:
            return data.item()
        return data.tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return str(value)


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON file using safe serialization."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
