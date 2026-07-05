from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class ForwardTurnCommand:
    """Mapped Task1 forward-turn command."""

    speed_factor: torch.Tensor
    forward_norm: torch.Tensor
    turn_norm: torch.Tensor
    wheel_norm: torch.Tensor


@dataclass(frozen=True)
class ForwardTurnProtocol:
    """Forward-throttle plus turn protocol used by Task1.

    The mapping is intentionally equivalent to the legacy Task1 formula. It is
    separated here to make the action protocol testable without launching
    IsaacLab.
    """

    min_forward_action: float
    max_forward_action: float
    forward_curve_power: float
    turn_scale_norm: float
    action_low: float = -1.0
    action_high: float = 1.0

    def sanitize(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.as_tensor(actions, dtype=torch.float32, device=actions.device if isinstance(actions, torch.Tensor) else None)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=self.action_high, neginf=self.action_low)
        return torch.clamp(actions, self.action_low, self.action_high)

    def map_to_normalized_wheels(self, actions: torch.Tensor) -> ForwardTurnCommand:
        clean_actions = self.sanitize(actions)
        if clean_actions.ndim != 2 or clean_actions.shape[-1] != 2:
            raise ValueError(f"ForwardTurnProtocol expects shape [N, 2], got {tuple(clean_actions.shape)}")

        forward_raw = clean_actions[:, 0]
        turn_raw = clean_actions[:, 1]

        speed_factor_linear = 0.5 * (forward_raw + 1.0)
        speed_factor = torch.pow(
            torch.clamp(speed_factor_linear, 0.0, 1.0),
            float(self.forward_curve_power),
        )
        forward_norm = float(self.min_forward_action) + (
            float(self.max_forward_action) - float(self.min_forward_action)
        ) * speed_factor
        forward_norm = torch.clamp(forward_norm, 0.0, 1.0)

        turn_norm = torch.clamp(turn_raw * float(self.turn_scale_norm), -1.0, 1.0)
        left_norm = torch.clamp(forward_norm - turn_norm, -1.0, 1.0)
        right_norm = torch.clamp(forward_norm + turn_norm, -1.0, 1.0)
        wheel_norm = torch.stack([left_norm, right_norm], dim=-1)

        return ForwardTurnCommand(
            speed_factor=speed_factor,
            forward_norm=forward_norm,
            turn_norm=turn_norm,
            wheel_norm=wheel_norm,
        )

    def map_to_wheel_velocity_targets(
        self,
        actions: torch.Tensor,
        wheel_speed_scale: float,
        wheel_signs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ForwardTurnCommand]:
        command = self.map_to_normalized_wheels(actions)
        wheel_targets = command.wheel_norm * float(wheel_speed_scale)
        if wheel_signs is not None:
            wheel_targets = wheel_targets * wheel_signs.to(device=wheel_targets.device, dtype=wheel_targets.dtype).unsqueeze(0)
        return wheel_targets, command
