from __future__ import annotations

import math
from typing import Union

import torch

TensorLike = Union[float, torch.Tensor]


def wrap_to_pi(angle: TensorLike) -> TensorLike:
    """Wrap an angle in radians to [-pi, pi]."""
    if isinstance(angle, torch.Tensor):
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def angle_difference(target_angle: TensorLike, source_angle: TensorLike) -> TensorLike:
    """Return target - source wrapped to [-pi, pi]."""
    return wrap_to_pi(target_angle - source_angle)  # type: ignore[operator]


def heading_error(target_xy: torch.Tensor, current_xy: torch.Tensor, current_yaw: torch.Tensor) -> torch.Tensor:
    """Compute wrapped heading error from current pose to target point."""
    delta = target_xy - current_xy
    target_yaw = torch.atan2(delta[..., 1], delta[..., 0])
    return wrap_to_pi(target_yaw - current_yaw)


def yaw_from_quat_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Return yaw from quaternion in xyzw order."""
    x = quat_xyzw[..., 0]
    y = quat_xyzw[..., 1]
    z = quat_xyzw[..., 2]
    w = quat_xyzw[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Return yaw from quaternion in wxyz order."""
    w = quat_wxyz[..., 0]
    x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]
    z = quat_wxyz[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)
