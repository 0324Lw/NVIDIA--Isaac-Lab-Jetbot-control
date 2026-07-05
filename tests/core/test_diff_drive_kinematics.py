from __future__ import annotations

import torch

from diff_drive_rl.core.physics.diff_drive_kinematics import (
    clamp_wheel_speed,
    compute_wheel_saturation_ratio,
    linear_angular_to_wheel,
    wheel_to_linear_angular,
)


def test_linear_angular_wheel_round_trip() -> None:
    linear = torch.tensor([-0.2, 0.0, 0.4, 1.0], dtype=torch.float32)
    angular = torch.tensor([0.5, -0.3, 0.0, 1.2], dtype=torch.float32)
    wheel_radius = 0.08
    wheel_base = 0.34

    left, right = linear_angular_to_wheel(linear, angular, wheel_radius, wheel_base)
    linear_back, angular_back = wheel_to_linear_angular(left, right, wheel_radius, wheel_base)

    assert torch.allclose(linear, linear_back, atol=1.0e-6)
    assert torch.allclose(angular, angular_back, atol=1.0e-6)


def test_clamp_wheel_speed_and_saturation_ratio() -> None:
    wheel_speed = torch.tensor([[-3.0, 0.0], [2.0, 4.0]], dtype=torch.float32)
    clamped = clamp_wheel_speed(wheel_speed, max_wheel_speed=2.0)

    assert torch.all(clamped <= 2.0)
    assert torch.all(clamped >= -2.0)
    assert torch.allclose(clamped, torch.tensor([[-2.0, 0.0], [2.0, 2.0]]))

    ratio = compute_wheel_saturation_ratio(clamped, max_wheel_speed=2.0)
    assert torch.isclose(ratio, torch.tensor(0.75))
