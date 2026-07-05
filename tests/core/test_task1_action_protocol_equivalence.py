from __future__ import annotations

import torch

from diff_drive_rl.core.physics.action_protocol import ForwardTurnProtocol


def legacy_task1_mapping(actions: torch.Tensor) -> torch.Tensor:
    actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    actions = torch.clamp(actions, -1.0, 1.0)
    forward_raw = actions[:, 0]
    turn_raw = actions[:, 1]
    speed_factor_linear = 0.5 * (forward_raw + 1.0)
    speed_factor = torch.pow(torch.clamp(speed_factor_linear, 0.0, 1.0), 2.0)
    forward_norm = 0.05 + (1.0 - 0.05) * speed_factor
    forward_norm = torch.clamp(forward_norm, 0.0, 1.0)
    turn_norm = torch.clamp(turn_raw * 0.85, -1.0, 1.0)
    left_norm = torch.clamp(forward_norm - turn_norm, -1.0, 1.0)
    right_norm = torch.clamp(forward_norm + turn_norm, -1.0, 1.0)
    return torch.stack([left_norm, right_norm], dim=-1)


def test_task1_forward_turn_mapping_matches_legacy_formula() -> None:
    protocol = ForwardTurnProtocol(
        min_forward_action=0.05,
        max_forward_action=1.0,
        forward_curve_power=2.0,
        turn_scale_norm=0.85,
    )
    actions = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.2, -1.0],
            [0.2, 1.0],
            [2.0, -2.0],
            [float("nan"), float("inf")],
            [float("-inf"), 0.5],
            [-0.37, 0.22],
        ],
        dtype=torch.float32,
    )

    expected = legacy_task1_mapping(actions)
    actual = protocol.map_to_normalized_wheels(actions).wheel_norm

    assert torch.allclose(actual, expected, atol=1.0e-7)


def test_task1_wheel_velocity_targets_apply_sign_and_scale() -> None:
    protocol = ForwardTurnProtocol(
        min_forward_action=0.05,
        max_forward_action=1.0,
        forward_curve_power=2.0,
        turn_scale_norm=0.85,
    )
    actions = torch.tensor([[0.0, 0.5], [0.3, -0.25]], dtype=torch.float32)
    wheel_signs = torch.tensor([1.0, -1.0], dtype=torch.float32)
    wheel_targets, command = protocol.map_to_wheel_velocity_targets(
        actions,
        wheel_speed_scale=16.0,
        wheel_signs=wheel_signs,
    )

    expected = legacy_task1_mapping(actions) * wheel_signs.unsqueeze(0) * 16.0
    assert torch.allclose(command.wheel_norm, legacy_task1_mapping(actions), atol=1.0e-7)
    assert torch.allclose(wheel_targets, expected, atol=1.0e-7)
