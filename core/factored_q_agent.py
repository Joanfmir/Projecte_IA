# core/factored_q_agent.py
"""
Agente Q-Learning factorizado con 3 Q-tables.
Cada Q-table maneja un contexto de decisión diferente.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import random
import pickle

from core.factored_states import FactoredStateEncoder
from core.dispatch_policy import (
    A_ASSIGN_URGENT_NEAREST,
    A_ASSIGN_ANY_NEAREST,
    A_WAIT,
    A_REPLAN_TRAFFIC,
)


@dataclass
class FactoredQConfig:
    """Configuración para el agente Q-Learning factorizado."""

    alpha: float = 0.15  # learning rate
    gamma: float = 0.95  # discount factor
    eps_start: float = 1.0  # epsilon inicial
    eps_min: float = 0.05  # epsilon mínimo
    eps_decay: float = 0.995  # decay por episodio
    rebal_interval: int = 10  # cada cuántos ticks usar Q₂


@dataclass
class FactoredQAgent:
    """
    Agente Q-Learning con 3 Q-tables factorizadas:
    - Q1 (Asignación): cuando hay pedidos sin asignar y riders elegibles
    - Q2 (Rebalanceo): cada K ticks para decisiones de balanceo
    - Q3 (Incidente): cuando cambia el tráfico
    """

    cfg: FactoredQConfig = field(default_factory=FactoredQConfig)
    encoder: FactoredStateEncoder = field(default_factory=FactoredStateEncoder)
    seed: int = 0

    # Q-tables: (state, action) -> value
    Q1: Dict[Tuple[Tuple, int], float] = field(default_factory=dict)  # Asignación
    Q2: Dict[Tuple[Tuple, int], float] = field(default_factory=dict)  # Rebalanceo
    Q3: Dict[Tuple[Tuple, int], float] = field(default_factory=dict)  # Incidente

    # Acciones por Q-table
    actions_q1: List[int] = field(
        default_factory=lambda: [A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT]
    )
    actions_q2: List[int] = field(
        default_factory=lambda: [A_ASSIGN_ANY_NEAREST, A_WAIT]
    )
    actions_q3: List[int] = field(default_factory=lambda: [A_REPLAN_TRAFFIC, A_WAIT])

    epsilon: float = field(init=False)
    rng: random.Random = field(init=False)

    # Para tracking
    last_q_used: str = field(default="none", init=False)
    max_delta_q: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.epsilon = self.cfg.eps_start
        self.rng = random.Random(self.seed)
        self.max_delta_q = 0.0

    # ─────────────────────────────────────────────────────────────
    # Getters/Setters de Q-values
    # ─────────────────────────────────────────────────────────────

    def get_q(self, q_table: Dict, state: Tuple, action: int) -> float:
        return q_table.get((state, action), 0.0)

    def set_q(self, q_table: Dict, state: Tuple, action: int, value: float) -> None:
        q_table[(state, action)] = value

    def best_action(self, q_table: Dict, state: Tuple, actions: List[int]) -> int:
        """Retorna la mejor acción según Q-values."""
        best_a = actions[0]
        best_v = self.get_q(q_table, state, best_a)
        for a in actions[1:]:
            v = self.get_q(q_table, state, a)
            if v > best_v:
                best_v = v
                best_a = a
        return best_a

    # ─────────────────────────────────────────────────────────────
    # Selección de acción
    # ─────────────────────────────────────────────────────────────

    def choose_action(self, snap: Dict, training: bool = True) -> int:
        """
        Decide qué acción tomar basado en el snapshot actual.
        Selecciona automáticamente cuál Q-table usar.
        """
        encoded = self.encoder.encode_all(snap)
        features = encoded["features"]
        t = features["t"]

        # Prioridad de Q-tables:
        # 1. Q₃ si hubo cambio de tráfico
        # 2. Q₁ si hay trabajo por asignar
        # 3. Q₂ si toca rebalanceo periódico
        # 4. Default: WAIT

        if self.encoder.should_use_q3(features):
            return self._choose_from_q3(encoded["s_incident"], training)

        if self.encoder.should_use_q1(features):
            return self._choose_from_q1(encoded["s_assign"], training)

        if self.encoder.should_use_q2(t, self.cfg.rebal_interval):
            return self._choose_from_q2(encoded["s_rebal"], training)

        # No hay nada que hacer
        self.last_q_used = "none"
        return A_WAIT

    def _choose_from_q1(self, state: Tuple, training: bool) -> int:
        """Elige acción desde Q₁ (Asignación)."""
        self.last_q_used = "Q1"
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions_q1)
        return self.best_action(self.Q1, state, self.actions_q1)

    def _choose_from_q2(self, state: Tuple, training: bool) -> int:
        """Elige acción desde Q₂ (Rebalanceo)."""
        self.last_q_used = "Q2"
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions_q2)
        return self.best_action(self.Q2, state, self.actions_q2)

    def _choose_from_q3(self, state: Tuple, training: bool) -> int:
        """Elige acción desde Q₃ (Incidente)."""
        self.last_q_used = "Q3"
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions_q3)
        return self.best_action(self.Q3, state, self.actions_q3)

    # ─────────────────────────────────────────────────────────────
    # Update (Q-Learning)
    # ─────────────────────────────────────────────────────────────

    def update(
        self, snap: Dict, action: int, reward: float, snap_next: Dict, done: bool
    ) -> None:
        """
        Actualiza la Q-table correspondiente según cuál se usó.
        """
        encoded = self.encoder.encode_all(snap)
        encoded_next = self.encoder.encode_all(snap_next)

        if self.last_q_used == "Q1":
            self._update_q1(
                encoded["s_assign"], action, reward, encoded_next["s_assign"], done
            )
        elif self.last_q_used == "Q2":
            self._update_q2(
                encoded["s_rebal"], action, reward, encoded_next["s_rebal"], done
            )
        elif self.last_q_used == "Q3":
            self._update_q3(
                encoded["s_incident"], action, reward, encoded_next["s_incident"], done
            )
        # Si last_q_used == "none", no actualizamos nada

    def _update_q_table(
        self,
        q_table: Dict,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        done: bool,
        actions: List[int],
    ) -> float:
        """Actualización Q-Learning estándar. Retorna delta Q."""
        q_old = self.get_q(q_table, state, action)

        if done:
            target = reward
        else:
            # max_a' Q(s', a')
            best_next = max(self.get_q(q_table, next_state, a) for a in actions)
            target = reward + self.cfg.gamma * best_next

        delta = abs(self.cfg.alpha * (target - q_old))
        q_new = q_old + self.cfg.alpha * (target - q_old)
        self.set_q(q_table, state, action, q_new)

        # Trackear máximo delta del episodio
        self.max_delta_q = max(self.max_delta_q, delta)
        return delta

    def _update_q1(self, s: Tuple, a: int, r: float, s2: Tuple, done: bool) -> None:
        self._update_q_table(self.Q1, s, a, r, s2, done, self.actions_q1)

    def _update_q2(self, s: Tuple, a: int, r: float, s2: Tuple, done: bool) -> None:
        self._update_q_table(self.Q2, s, a, r, s2, done, self.actions_q2)

    def _update_q3(self, s: Tuple, a: int, r: float, s2: Tuple, done: bool) -> None:
        self._update_q_table(self.Q3, s, a, r, s2, done, self.actions_q3)

    # ─────────────────────────────────────────────────────────────
    # Epsilon decay
    # ─────────────────────────────────────────────────────────────

    def decay_epsilon(self) -> None:
        """Reduce epsilon (llamar al final de cada episodio)."""
        self.epsilon = max(self.cfg.eps_min, self.epsilon * self.cfg.eps_decay)

    # ─────────────────────────────────────────────────────────────
    # Persistencia
    # ─────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Guarda las Q-tables y configuración."""
        payload = {
            "Q1": self.Q1,
            "Q2": self.Q2,
            "Q3": self.Q3,
            "epsilon": self.epsilon,
            "cfg": self.cfg,
            "actions_q1": self.actions_q1,
            "actions_q2": self.actions_q2,
            "actions_q3": self.actions_q3,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str, episode_len: int = 900) -> "FactoredQAgent":
        """Carga un agente desde archivo."""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        agent = FactoredQAgent(
            cfg=payload["cfg"],
            encoder=FactoredStateEncoder(episode_len=episode_len),
        )
        agent.Q1 = payload["Q1"]
        agent.Q2 = payload["Q2"]
        agent.Q3 = payload["Q3"]
        agent.epsilon = payload.get("epsilon", payload["cfg"].eps_min)
        agent.actions_q1 = payload.get("actions_q1", agent.actions_q1)
        agent.actions_q2 = payload.get("actions_q2", agent.actions_q2)
        agent.actions_q3 = payload.get("actions_q3", agent.actions_q3)
        return agent

    # ─────────────────────────────────────────────────────────────
    # Estadísticas
    # ─────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del agente."""
        return {
            "Q1_entries": len(self.Q1),
            "Q2_entries": len(self.Q2),
            "Q3_entries": len(self.Q3),
            "total_entries": len(self.Q1) + len(self.Q2) + len(self.Q3),
            "epsilon": self.epsilon,
            "max_delta_q": self.max_delta_q,
        }

    def reset_delta(self) -> None:
        """Resetea max_delta_q. Llamar al inicio de cada episodio."""
        self.max_delta_q = 0.0

    def get_max_delta(self) -> float:
        """Retorna el máximo delta Q del episodio actual."""
        return self.max_delta_q


# ─────────────────────────────────────────────────────────────
# Test básico
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test de instanciación
    agent = FactoredQAgent(seed=42)

    # Simular un snapshot básico
    fake_snap = {
        "t": 100,
        "restaurant": (22, 17),
        "pending_orders": [
            ((10, 5), 1, 150, None),  # (dropoff, priority, deadline, assigned_to)
            ((30, 20), 2, 120, None),
        ],
        "riders": [
            {
                "id": 1,
                "pos": (22, 17),
                "available": True,
                "assigned": [],
                "route": [],
                "fatigue": 0.5,
            },
            {
                "id": 2,
                "pos": (15, 10),
                "available": False,
                "assigned": [1],
                "route": [(16, 10)],
                "fatigue": 1.2,
            },
        ],
        "traffic_zones": {0: "low", 1: "medium", 2: "high", 3: "low"},
        "traffic": "medium",
    }

    # Test choose_action
    action = agent.choose_action(fake_snap, training=True)
    print(f"Acción elegida: {action} (Q usada: {agent.last_q_used})")

    # Test stats
    print(f"Stats: {agent.stats()}")
