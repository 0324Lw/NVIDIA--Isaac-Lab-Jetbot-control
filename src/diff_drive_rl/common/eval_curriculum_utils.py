from __future__ import annotations


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def stage_from_progress(k: float, boundaries=(0.12, 0.28, 0.45, 0.65, 0.82)) -> int:
    k = clamp01(k)
    for idx, boundary in enumerate(boundaries):
        if k < boundary:
            return idx
    return len(boundaries)
