"""
Heuristic policy operating on simulator snapshots.
Provides the same interface as FactoredQAgent for visualization and runners.
"""
from __future__ import annotations

from core.shared_params import A_ASSIGN_ANY_NEAREST


class HeuristicPolicy:
    """Simple baseline policy that always assigns nearest available rider."""

    def choose_action_snapshot(self, _snap: dict) -> int:
        return A_ASSIGN_ANY_NEAREST


__all__ = ["HeuristicPolicy"]
