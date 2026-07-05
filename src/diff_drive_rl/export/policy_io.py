from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from diff_drive_rl.training.checkpoint_utils import json_safe, save_json


def build_policy_io(
    *,
    task_name: str,
    actor_obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
    action_protocol: str,
    observation_protocol: str,
    model_protocol: str,
    control_dt: float,
    frame_stack: int,
    normalizer_source: str = "actor_obs_norm",
    onnx_export_target: str = "actor_only",
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build policy IO metadata for actor-only export and sim2sim checks."""
    payload: Dict[str, Any] = {
        "task_name": str(task_name),
        "actor_obs_dim": int(actor_obs_dim),
        "critic_obs_dim": int(critic_obs_dim),
        "action_dim": int(action_dim),
        "action_protocol": str(action_protocol),
        "observation_protocol": str(observation_protocol),
        "model_protocol": str(model_protocol),
        "control_dt": float(control_dt),
        "frame_stack": int(frame_stack),
        "normalizer_source": str(normalizer_source),
        "onnx_export_target": str(onnx_export_target),
    }
    if extra:
        payload["extra"] = json_safe(dict(extra))
    return json_safe(payload)


def write_policy_io(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write policy IO metadata to disk."""
    save_json(path, payload)


def load_policy_io(path: str | Path) -> Dict[str, Any]:
    """Load policy IO metadata."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_policy_io(payload: Mapping[str, Any]) -> None:
    """Validate required policy IO fields."""
    required = [
        "task_name",
        "actor_obs_dim",
        "critic_obs_dim",
        "action_dim",
        "action_protocol",
        "observation_protocol",
        "model_protocol",
        "control_dt",
        "frame_stack",
        "normalizer_source",
        "onnx_export_target",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"policy_io missing required fields: {missing}")
    for key in ("actor_obs_dim", "critic_obs_dim", "action_dim", "frame_stack"):
        if int(payload[key]) <= 0:
            raise ValueError(f"policy_io field {key} must be positive")
    if str(payload["onnx_export_target"]) != "actor_only":
        raise ValueError("Only actor_only export metadata is supported by this framework")
