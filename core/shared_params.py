"""
Shared action constants for the factored agent stack and heuristic policy.
"""
from __future__ import annotations

# Available actions (consumed by simulator, agents, and tests)
A_ASSIGN_URGENT_NEAREST = 0
A_ASSIGN_ANY_NEAREST = 1
A_WAIT = 2
A_REPLAN_TRAFFIC = 3

# Full action set reference
ALL_ACTIONS = [
    A_ASSIGN_URGENT_NEAREST,
    A_ASSIGN_ANY_NEAREST,
    A_WAIT,
    A_REPLAN_TRAFFIC,
]

__all__ = [
    "A_ASSIGN_URGENT_NEAREST",
    "A_ASSIGN_ANY_NEAREST",
    "A_WAIT",
    "A_REPLAN_TRAFFIC",
    "ALL_ACTIONS",
]
