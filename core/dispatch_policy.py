# core/dispatch_policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import random
import math

State = Tuple[int, int, int, int, int, int, int, int]  # bins
Action = int

A_ASSIGN_URGENT_NEAREST = 0
A_ASSIGN_ANY_NEAREST    = 1
A_WAIT                  = 2
A_REPLAN_TRAFFIC        = 3

ALL_ACTIONS = [A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC]

@dataclass
class QLearningConfig:
    alpha: float = 0.15
    gamma: float = 0.95
    epsilon: float = 0.20
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995


class DispatchPolicy:
    """
    Política RL (Q-learning tabular) que decide la estrategia.
    La micro-asignación la hace AssignmentEngine.
    """

    def __init__(self, cfg: QLearningConfig, seed: int = 123):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.Q: Dict[Tuple[State, Action], float] = {}

    def get_Q(self, s: State, a: Action) -> float:
        return self.Q.get((s, a), 0.0)

    def set_Q(self, s: State, a: Action, v: float) -> None:
        self.Q[(s, a)] = v

    def choose_action(self, s: State) -> Action:
        # epsilon-greedy
        if self.rng.random() < self.cfg.epsilon:
            return self.rng.choice(ALL_ACTIONS)

        # greedy
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


# -----------------------------
# Discretización del estado
# -----------------------------

def bin_time(t: int, episode_len: int) -> int:
    # 0..4
    frac = t / max(1, episode_len)
    if frac < 0.2: return 0
    if frac < 0.4: return 1
    if frac < 0.6: return 2
    if frac < 0.8: return 3
    return 4

def bin_pending(n: int) -> int:
    # 0, 1-2, 3-5, 6-10, >10
    if n == 0: return 0
    if n <= 2: return 1
    if n <= 5: return 2
    if n <= 10: return 3
    return 4

def bin_ratio_urgent(r: float) -> int:
    # 0, 1-25, 26-50, 51-75, 76-100
    if r <= 0.0: return 0
    if r <= 0.25: return 1
    if r <= 0.50: return 2
    if r <= 0.75: return 3
    return 4

def bin_free_riders(n: int) -> int:
    if n == 0: return 0
    if n == 1: return 1
    if n == 2: return 2
    return 3  # 3+

def bin_fatigue(avg: float) -> int:
    # (ajústalo luego). v1: fatiga crece ~0.05 por movimiento
    if avg < 1.0: return 0
    if avg < 2.5: return 1
    return 2

def bin_imbalance(std_deliveries: float) -> int:
    if std_deliveries < 0.5: return 0
    if std_deliveries < 1.5: return 1
    return 2

def bin_traffic(level: str) -> int:
    # low/medium/high -> 0/1/2
    return {"low": 0, "medium": 1, "high": 2}.get(level, 0)

def bin_closures(n: int) -> int:
    if n == 0: return 0
    if n == 1: return 1
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
    closures: int
) -> State:
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
