# core/dispatch_policy.py
from __future__ import annotations
import random
import warnings
from typing import Dict, Tuple

from core.shared_params import (  # re-exported legacy constants/helpers
    A_ASSIGN_ANY_NEAREST,
    A_ASSIGN_URGENT_NEAREST,
    A_REPLAN_TRAFFIC,
    A_WAIT,
    ALL_ACTIONS,
    QLearningConfig,
    bin_closures,
    bin_fatigue,
    bin_free_riders,
    bin_imbalance,
    bin_pending,
    bin_ratio_urgent,
    bin_time,
    bin_traffic,
    make_state,
)

State = Tuple[int, int, int, int, int, int, int, int]  # bins
Action = int


class DispatchPolicy:
    """
    Política RL (Q-learning tabular) legacy. ⚠️ Deprecated.

    Se mantiene como shim para compatibilidad; use FactoredQAgent.
    """

    def __init__(self, cfg: QLearningConfig, seed: int = 123):
        warnings.warn(
            "DispatchPolicy está deprecado; usa FactoredQAgent o core.shared_params",
            DeprecationWarning,
            stacklevel=2,
        )
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.Q: Dict[Tuple[State, Action], float] = {}

    def get_Q(self, s: State, a: Action) -> float:
        return self.Q.get((s, a), 0.0)

    def set_Q(self, s: State, a: Action, v: float) -> None:
        self.Q[(s, a)] = v

    def choose_action(self, s: State) -> Action:
        if self.rng.random() < self.cfg.epsilon:
            return self.rng.choice(ALL_ACTIONS)

        qs = [(self.get_Q(s, a), a) for a in ALL_ACTIONS]
        qs.sort(reverse=True, key=lambda x: x[0])
        return qs[0][1]

    def update(self, s: State, a: Action, r: float, s2: State, done: bool) -> None:
        old = self.get_Q(s, a)
        if done:
            target = r
        else:
            best_next = max(self.get_Q(s2, a2) for a2 in ALL_ACTIONS)
            target = r + self.cfg.gamma * best_next
        new = old + self.cfg.alpha * (target - old)
        self.set_Q(s, a, new)

    def decay_epsilon(self) -> None:
        self.cfg.epsilon = max(self.cfg.epsilon_min, self.cfg.epsilon * self.cfg.epsilon_decay)


__all__ = [
    "A_ASSIGN_URGENT_NEAREST",
    "A_ASSIGN_ANY_NEAREST",
    "A_WAIT",
    "A_REPLAN_TRAFFIC",
    "ALL_ACTIONS",
    "QLearningConfig",
    "DispatchPolicy",
    "bin_time",
    "bin_pending",
    "bin_ratio_urgent",
    "bin_free_riders",
    "bin_fatigue",
    "bin_imbalance",
    "bin_traffic",
    "bin_closures",
    "make_state",
]
