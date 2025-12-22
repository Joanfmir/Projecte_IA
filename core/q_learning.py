# core/q_learning.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import random
import pickle


@dataclass
class QConfig:
    alpha: float = 0.15     # learning rate
    gamma: float = 0.95     # discount
    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay: float = 0.995


class QLearningAgent:
    def __init__(self, actions: List[int], cfg: QConfig, seed: int = 0):
        self.actions = list(actions)
        self.cfg = cfg
        self.rng = random.Random(seed)

        self.epsilon = cfg.eps_start
        # Q[(state, action)] = value
        self.Q: Dict[Tuple[Any, int], float] = {}

    def get_q(self, s: Any, a: int) -> float:
        return self.Q.get((s, a), 0.0)

    def best_action(self, s: Any) -> int:
        best_a = self.actions[0]
        best_v = self.get_q(s, best_a)
        for a in self.actions[1:]:
            v = self.get_q(s, a)
            if v > best_v:
                best_v = v
                best_a = a
        return best_a

    def choose_action(self, s: Any, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.best_action(s)

    def update(self, s: Any, a: int, r: float, s2: Any, done: bool) -> None:
        q = self.get_q(s, a)

        if done:
            target = r
        else:
            # max_a' Q(s',a')
            best_next = max(self.get_q(s2, ap) for ap in self.actions)
            target = r + self.cfg.gamma * best_next

        new_q = q + self.cfg.alpha * (target - q)
        self.Q[(s, a)] = new_q

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.eps_min, self.epsilon * self.cfg.eps_decay)

    def save(self, path: str) -> None:
        payload = {
            "Q": self.Q,
            "epsilon": self.epsilon,
            "actions": self.actions,
            "cfg": self.cfg,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str) -> "QLearningAgent":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        agent = QLearningAgent(payload["actions"], payload["cfg"])
        agent.Q = payload["Q"]
        agent.epsilon = payload.get("epsilon", payload["cfg"].eps_min)
        return agent
