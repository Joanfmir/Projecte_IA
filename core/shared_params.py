"""
Shared constants and discretization helpers used across simulator, agents and
visualization. This is the single source of truth for action ids and the
legacy Q-learning defaults consumed by both the factored agent and legacy
shims.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

State = Tuple[int, int, int, int, int, int, int, int]  # bins legacy

# Available actions (consumed by simulator, agents, and tests)
A_ASSIGN_URGENT_NEAREST = 0
A_ASSIGN_ANY_NEAREST = 1
A_WAIT = 2
A_REPLAN_TRAFFIC = 3

# Full legacy/tabular action set
ALL_ACTIONS = [
    A_ASSIGN_URGENT_NEAREST,
    A_ASSIGN_ANY_NEAREST,
    A_WAIT,
    A_REPLAN_TRAFFIC,
]


@dataclass
class QLearningConfig:
    """Default parameters for the legacy Q-learning agent."""

    alpha: float = 0.15
    gamma: float = 0.95
    epsilon: float = 0.20
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995


# -----------------------------
# Discretización del estado
# -----------------------------


def bin_time(t: int, episode_len: int) -> int:
    """Bucket for normalized time (consumed by simulator.make_state)."""
    frac = t / max(1, episode_len)
    if frac < 0.2:
        return 0
    if frac < 0.4:
        return 1
    if frac < 0.6:
        return 2
    if frac < 0.8:
        return 3
    return 4


def bin_pending(n: int) -> int:
    """Bucket for pending orders."""
    if n == 0:
        return 0
    if n <= 2:
        return 1
    if n <= 5:
        return 2
    if n <= 10:
        return 3
    return 4


def bin_ratio_urgent(r: float) -> int:
    """Bucket for urgent-order ratio."""
    if r <= 0.0:
        return 0
    if r <= 0.25:
        return 1
    if r <= 0.50:
        return 2
    if r <= 0.75:
        return 3
    return 4


def bin_free_riders(n: int) -> int:
    """Bucket for free riders."""
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    return 3  # 3+


def bin_fatigue(avg: float) -> int:
    """Bucket for average fatigue."""
    if avg < 1.0:
        return 0
    if avg < 2.5:
        return 1
    return 2


def bin_imbalance(std_deliveries: float) -> int:
    """Bucket for imbalance of deliveries across riders."""
    if std_deliveries < 0.5:
        return 0
    if std_deliveries < 1.5:
        return 1
    return 2


def bin_traffic(level: str) -> int:
    """Bucket for global traffic level."""
    return {"low": 0, "medium": 1, "high": 2}.get(level, 0)


def bin_closures(n: int) -> int:
    """Bucket for active road closures."""
    if n == 0:
        return 0
    if n == 1:
        return 1
    return 2


def make_state(
    t: int,
    episode_len: int,
    pending: int,
    urgent_ratio: float,
    free_riders: int,
    avg_fatigue: float,
    std_deliveries: float,
    traffic_level: str,
    closures: int,
) -> State:
    """Build the legacy discretized state used by the simulator."""
    return (
        bin_time(t, episode_len),
        bin_pending(pending),
        bin_ratio_urgent(urgent_ratio),
        bin_free_riders(free_riders),
        bin_fatigue(avg_fatigue),
        bin_imbalance(std_deliveries),
        bin_traffic(traffic_level),
        bin_closures(closures),
    )


__all__ = [
    "A_ASSIGN_URGENT_NEAREST",
    "A_ASSIGN_ANY_NEAREST",
    "A_WAIT",
    "A_REPLAN_TRAFFIC",
    "ALL_ACTIONS",
    "QLearningConfig",
    "bin_time",
    "bin_pending",
    "bin_ratio_urgent",
    "bin_free_riders",
    "bin_fatigue",
    "bin_imbalance",
    "bin_traffic",
    "bin_closures",
    "make_state",
    "State",
]
