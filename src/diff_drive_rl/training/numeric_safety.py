from __future__ import annotations

from typing import Any

import torch


def sanitize_tensor(value: torch.Tensor, nan: float = 0.0, posinf: float = 1.0, neginf: float = -1.0) -> torch.Tensor:
    """Replace NaN and infinite values in a tensor."""
    return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))


def sanitize_action(action: torch.Tensor, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    """Sanitize and clamp policy actions."""
    action = sanitize_tensor(action, nan=0.0, posinf=float(high), neginf=float(low))
    return torch.clamp(action, float(low), float(high))


def contains_nan_or_inf(value: torch.Tensor) -> bool:
    """Return True if a tensor contains NaN or infinite values."""
    return bool((~torch.isfinite(value)).any().item())


def safe_mean(value: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    """Compute a finite mean and return default for empty tensors."""
    if value.numel() == 0:
        return torch.as_tensor(float(default), dtype=torch.float32, device=value.device)
    value = sanitize_tensor(value, nan=float(default), posinf=float(default), neginf=float(default))
    return value.mean()


def safe_item(value: Any, default: float = 0.0) -> float:
    """Convert tensor-like values to a finite Python float."""
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return float(default)
            value = value.detach().float().mean().cpu().item()
        out = float(value)
        if out != out or out == float("inf") or out == float("-inf"):
            return float(default)
        return out
    except Exception:
        return float(default)
