# core/factored_q_agent.py
"""
Agente Q-Learning factorizado con 2 Q-tables: Q1 (asignación) y Q3 (incidente/tráfico).
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


@dataclass
class FactoredQAgent:
    """
    Agente Q-Learning con 2 Q-tables factorizadas:
    - Q1 (Asignación): cuando hay pedidos sin asignar y riders elegibles
    - Q3 (Incidente): cuando cambia el tráfico significativamente
    """

    cfg: FactoredQConfig = field(default_factory=FactoredQConfig)
    encoder: FactoredStateEncoder = field(default_factory=FactoredStateEncoder)
    seed: int = 0

    # Q-tables: (state, action) -> value
    Q1: Dict[Tuple[Tuple, int], float] = field(default_factory=dict)  # Asignación
    Q3: Dict[Tuple[Tuple, int], float] = field(default_factory=dict)  # Incidente

    # Acciones por Q-table (referencia, pero usamos masking dinámico)
    actions_q1: List[int] = field(
        default_factory=lambda: [A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT]
    )
    actions_q3: List[int] = field(default_factory=lambda: [A_REPLAN_TRAFFIC, A_WAIT])

    epsilon: float = field(init=False)
    rng: random.Random = field(init=False)

    # Para tracking
    last_q_used: str = field(default="none", init=False)
    max_delta_q: float = field(default=0.0, init=False)
    last_action: Optional[int] = field(default=None, init=False)

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
        """Retorna la mejor acción con tie-break aleatorio."""
        q_values = [(a, self.get_q(q_table, state, a)) for a in actions]
        max_q = max(v for _, v in q_values)
        best_actions = [a for a, v in q_values if v == max_q]
        # Deterministic tie-break by action id
        return sorted(best_actions)[0]

    # ─────────────────────────────────────────────────────────────
    # Action Masking - acciones válidas por tabla
    # ─────────────────────────────────────────────────────────────

    def _valid_actions_q1(self, features: Dict) -> List[int]:
        """Filtra acciones inválidas para Q1 (asignación)."""
        valid = []
        has_urgent = features["urgent_unassigned"] > 0
        has_pending = features["pending_unassigned"] > 0
        has_free = features["free_riders"] > 0

        if has_pending and has_free:
            if has_urgent:
                valid.append(A_ASSIGN_URGENT_NEAREST)
            valid.append(A_ASSIGN_ANY_NEAREST)
        valid.append(A_WAIT)  # Siempre válido como fallback
        return valid

    def _valid_actions_q3(self, features: Dict) -> List[int]:
        """Filtra acciones inválidas para Q3 (incidente/tráfico)."""
        has_busy = features["busy_riders"] > 0
        if has_busy:
            return [A_REPLAN_TRAFFIC, A_WAIT]
        return [A_WAIT]

    # ─────────────────────────────────────────────────────────────
    # Selección de acción
    # ─────────────────────────────────────────────────────────────

    def choose_action(self, snap: Dict, training: bool = True) -> int:
        """
        Decide qué acción tomar basado en el snapshot actual.

        PRIORIDAD:
        1. Q1 si hay trabajo (pending > 0 AND free_riders > 0)
        2. Q3 si cambió tráfico Y hay riders en ruta (para replan)
        3. WAIT si no hay nada que hacer
        """
        encoded = self.encoder.encode_all(snap, update_prev=False)
        features = encoded["features"]

        has_work = features["pending_unassigned"] > 0 and features["free_riders"] > 0
        has_riders_in_route = features["busy_riders"] > 0
        traffic_changed = self.encoder.should_use_q3(features)

        # PRIORIDAD 1: Siempre asignar si hay trabajo
        if has_work:
            action = self._choose_from_q1(encoded["s_assign"], features, training)
            self.last_action = action
            return action

        # PRIORIDAD 2: Replanificar solo si cambió tráfico Y hay riders activos
        if traffic_changed and has_riders_in_route:
            action = self._choose_from_q3(encoded["s_incident"], features, training)
            self.last_action = action
            return action

        # Sin trabajo, esperar
        self.last_q_used = "none"
        self.last_action = A_WAIT
        return A_WAIT

    def _choose_from_q1(self, state: Tuple, features: Dict, training: bool) -> int:
        """Elige acción desde Q₁ (Asignación) con action masking."""
        self.last_q_used = "Q1"
        valid = self._valid_actions_q1(features)
        if training and self.rng.random() < self.epsilon:
            action = self.rng.choice(valid)
        else:
            action = self.best_action(self.Q1, state, valid)
        self.last_action = action
        return action

    def _choose_from_q3(self, state: Tuple, features: Dict, training: bool) -> int:
        """Elige acción desde Q₃ (Incidente) con action masking."""
        self.last_q_used = "Q3"
        valid = self._valid_actions_q3(features)
        if training and self.rng.random() < self.epsilon:
            action = self.rng.choice(valid)
        else:
            action = self.best_action(self.Q3, state, valid)
        self.last_action = action
        return action

    # ─────────────────────────────────────────────────────────────
    # Update (Q-Learning con Unified Value Estimation + Masking)
    # ─────────────────────────────────────────────────────────────

    def get_max_q_for_next_state(self, snap_next: Dict) -> float:
        """
        Calcula max Q del estado siguiente usando la tabla que estará activa en t+1.
        Aplica action masking: solo considera acciones válidas en s'.

        Returns:
            float: max_a Q(s', a) de la tabla aplicable en s', sobre acciones válidas.
        """
        encoded = self.encoder.encode_all(snap_next, update_prev=False)
        features = encoded["features"]

        has_work = features["pending_unassigned"] > 0 and features["free_riders"] > 0
        has_riders_in_route = features["busy_riders"] > 0
        traffic_changed = self.encoder.should_use_q3(features)

        # Misma lógica que choose_action + action masking
        if has_work:
            state = encoded["s_assign"]
            valid = self._valid_actions_q1(features)
            if not valid:  # Fallback de seguridad
                return 0.0
            return max(self.get_q(self.Q1, state, a) for a in valid)

        if traffic_changed and has_riders_in_route:
            state = encoded["s_incident"]
            valid = self._valid_actions_q3(features)
            if not valid:  # Fallback de seguridad
                return 0.0
            return max(self.get_q(self.Q3, state, a) for a in valid)

        # Sin trabajo ni replan, valor terminal
        return 0.0

    def update(
        self, snap: Dict, action: int, reward: float, snap_next: Dict, done: bool
    ) -> None:
        """
        Actualiza la Q-table correspondiente usando Unified Value Estimation.
        Target = reward + gamma * get_max_q_for_next_state(snap_next)
        """
        # Codificar estado actual
        encoded = self.encoder.encode_all(snap, update_prev=False)
        features = encoded["features"]

        table_to_use = self.last_q_used
        state = None
        q_table = None

        if table_to_use == "none":
            # Inferir tabla aplicable (permite aprender con WAIT)
            has_work = features["pending_unassigned"] > 0
            has_riders_in_route = features["busy_riders"] > 0
            traffic_changed = self.encoder.should_use_q3(features)

            if has_work:
                table_to_use = "Q1"
                state = encoded["s_assign"]
                q_table = self.Q1
            elif traffic_changed and has_riders_in_route:
                table_to_use = "Q3"
                state = encoded["s_incident"]
                q_table = self.Q3
        else:
            if table_to_use == "Q1":
                state = encoded["s_assign"]
                q_table = self.Q1
            elif table_to_use == "Q3":
                state = encoded["s_incident"]
                q_table = self.Q3

        if state is None or q_table is None:
            return

        # Calcular target con Unified Value Estimation
        if done:
            target = reward
        else:
            max_q_next = self.get_max_q_for_next_state(snap_next)
            target = reward + self.cfg.gamma * max_q_next

        self.last_q_used = table_to_use
        self._unified_update(q_table, state, action, target)

    def _unified_update(
        self, q_table: Dict, state: Tuple, action: int, target: float
    ) -> None:
        """Actualización Q-Learning con target pre-calculado."""
        q_old = self.get_q(q_table, state, action)
        delta = abs(self.cfg.alpha * (target - q_old))
        q_new = q_old + self.cfg.alpha * (target - q_old)
        self.set_q(q_table, state, action, q_new)

        # Tracking
        if delta > self.max_delta_q:
            self.max_delta_q = delta

    def commit_encoder(self, snap: Dict) -> None:
        """
        Actualiza prev_traffic_pressure del encoder.
        Llamar EXACTAMENTE una vez por tick, después del update(), con snap (pre-step).
        """
        self.encoder.commit(snap)

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
            "Q3": self.Q3,
            "epsilon": self.epsilon,
            "cfg": self.cfg,
            "actions_q1": self.actions_q1,
            "actions_q3": self.actions_q3,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str, episode_len: int = 900) -> "FactoredQAgent":
        """Carga un agente desde archivo. Migra archivos antiguos con Q2."""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        agent = FactoredQAgent(
            cfg=payload["cfg"],
            encoder=FactoredStateEncoder(episode_len=episode_len),
        )
        agent.Q1 = payload["Q1"]
        agent.Q3 = payload["Q3"]
        agent.epsilon = payload.get("epsilon", payload["cfg"].eps_min)
        agent.actions_q1 = payload.get("actions_q1", agent.actions_q1)
        agent.actions_q3 = payload.get("actions_q3", agent.actions_q3)
        # Nota: Q2 ignorado si existe en archivos antiguos
        return agent

    # ─────────────────────────────────────────────────────────────
    # Estadísticas
    # ─────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del agente."""
        return {
            "Q1_entries": len(self.Q1),
            "Q3_entries": len(self.Q3),
            "total_entries": len(self.Q1) + len(self.Q3),
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
