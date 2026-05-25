from __future__ import annotations

import torch
import torch.nn as nn


def mlp(sizes, activation=nn.ELU, output_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(int(sizes[i]), int(sizes[i + 1])))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())
    return nn.Sequential(*layers)


def sanitize_tensor(x: torch.Tensor, clip_abs: float = 10.0) -> torch.Tensor:
    return torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=clip_abs, neginf=-clip_abs), -clip_abs, clip_abs)
