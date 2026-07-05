from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diff_drive_rl.export.policy_io import build_policy_io, load_policy_io, validate_policy_io, write_policy_io


def test_policy_io_serializes_tensor_numpy_and_path(tmp_path: Path) -> None:
    payload = build_policy_io(
        task_name="task1_multi_waypoint_navigation",
        actor_obs_dim=42,
        critic_obs_dim=42,
        action_dim=2,
        action_protocol="forward_throttle_turn_v1",
        observation_protocol="CoreNav-v1",
        model_protocol="ModularActor-v1",
        control_dt=1.0 / 48.0,
        frame_stack=3,
        extra={
            "tensor_scalar": torch.tensor(1.25),
            "tensor_vector": torch.tensor([1.0, 2.0]),
            "numpy_array": np.array([3.0, 4.0], dtype=np.float32),
            "path": tmp_path,
        },
    )

    path = tmp_path / "policy_io.json"
    write_policy_io(path, payload)
    loaded = load_policy_io(path)
    validate_policy_io(loaded)

    assert loaded["actor_obs_dim"] == 42
    assert loaded["critic_obs_dim"] == 42
    assert loaded["action_dim"] == 2
    assert loaded["onnx_export_target"] == "actor_only"
    assert loaded["normalizer_source"] == "actor_obs_norm"
    assert loaded["extra"]["tensor_scalar"] == 1.25
    assert loaded["extra"]["tensor_vector"] == [1.0, 2.0]
    assert loaded["extra"]["numpy_array"] == [3.0, 4.0]


def test_policy_io_validation_rejects_missing_fields() -> None:
    try:
        validate_policy_io({"task_name": "missing_fields"})
    except ValueError as exc:
        assert "missing required" in str(exc)
    else:
        raise AssertionError("validate_policy_io should reject missing fields")
