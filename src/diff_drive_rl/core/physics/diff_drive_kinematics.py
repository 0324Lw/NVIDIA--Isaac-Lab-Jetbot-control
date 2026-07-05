from __future__ import annotations

from typing import Tuple

import torch


def linear_angular_to_wheel(
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    wheel_radius: float,
    wheel_base: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert chassis linear/angular velocity to left/right wheel angular velocity."""
    radius = max(float(wheel_radius), 1.0e-8)
    base = float(wheel_base)
    left = (linear_velocity - 0.5 * angular_velocity * base) / radius
    right = (linear_velocity + 0.5 * angular_velocity * base) / radius
    return left, right


def wheel_to_linear_angular(
    left_wheel_velocity: torch.Tensor,
    right_wheel_velocity: torch.Tensor,
    wheel_radius: float,
    wheel_base: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert left/right wheel angular velocity to chassis linear/angular velocity."""
    radius = float(wheel_radius)
    base = max(float(wheel_base), 1.0e-8)
    linear = 0.5 * radius * (left_wheel_velocity + right_wheel_velocity)
    angular = radius * (right_wheel_velocity - left_wheel_velocity) / base
    return linear, angular


def clamp_wheel_speed(wheel_speed: torch.Tensor, max_wheel_speed: float) -> torch.Tensor:
    """Clamp wheel angular velocity to symmetric speed limits."""
    limit = abs(float(max_wheel_speed))
    if limit <= 0.0:
        return torch.zeros_like(wheel_speed)
    return torch.clamp(wheel_speed, -limit, limit)


def compute_wheel_saturation_ratio(wheel_speed: torch.Tensor, max_wheel_speed: float, eps: float = 1.0e-6) -> torch.Tensor:
    """Return the fraction of wheel commands at or beyond the speed limit."""
    limit = abs(float(max_wheel_speed))
    if limit <= 0.0:
        return torch.zeros((), dtype=wheel_speed.dtype, device=wheel_speed.device)
    saturated = torch.abs(wheel_speed) >= (limit - float(eps))
    return saturated.to(dtype=torch.float32).mean()
