from __future__ import annotations

from typing import Any, Dict


def format_metric_value(value: Any) -> str:
    try:
        value = float(value)
        if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
            return f"{value:.3e}"
        return f"{value:.6f}"
    except Exception:
        return str(value)


def make_table(title: str, data: Dict[str, Any], width: int = 120) -> str:
    lines = ["-" * width, f"| {title:<{width - 4}} |", "-" * width]
    if not data:
        lines.append(f"| {'<empty>':<{width - 4}} |")
    else:
        for key in sorted(data.keys()):
            k = str(key)
            v = format_metric_value(data[key])
            lines.append(f"| {k[:75]:<75} | {v[:35]:>35} |")
    lines.append("-" * width)
    return "\n".join(lines)
