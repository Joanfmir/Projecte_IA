# core/q_learning.py
"""Implementación genérica de un agente Q-Learning tabular.

Este módulo provee una clase base `QLearningAgent` y su configuración `QConfig`.
Encapsula la lógica estándar del algoritmo Q-Learning: mantenimiento de la tabla Q,
selección de acciones (epsilon-greedy) y actualización de valores Q.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import random
import pickle


@dataclass
class QConfig:
    """Configuración de hiperparámetros para el agente Q-Learning.

    Attributes:
        alpha: Tasa de aprendizaje (learning rate).
        gamma: Factor de descuento (discount factor).
        eps_start: Epsilon inicial para la política epsilon-greedy.
        eps_min: Valor mínimo de epsilon.
        eps_decay: Factor de decaimiento de epsilon por episodio.
    """
    alpha: float = 0.15
    gamma: float = 0.95
    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay: float = 0.995


class QLearningAgent:
    """Agente Q-Learning tabular estándar.

    Mantiene una tabla Q mapeando (estado, acción) -> valor y permite
    entrenamiento mediante exploración epsilon-greedy.

    Attributes:
        actions: Lista de acciones posibles.
        cfg: Configuración de hiperparámetros.
        rng: Generador de números aleatorios.
        epsilon: Valor actual de epsilon.
        Q: Diccionario que almacena los valores Q {(estado, acción): valor}.
    """

    def __init__(self, actions: List[int], cfg: QConfig, seed: int = 0):
        self.actions = list(actions)
        self.cfg = cfg
        self.rng = random.Random(seed)

        self.epsilon = cfg.eps_start
        # Q[(state, action)] = value
        self.Q: Dict[Tuple[Any, int], float] = {}

    def get_q(self, s: Any, a: int) -> float:
        """Obtiene el valor Q para un par estado-acción. Devuelve 0.0 si no existe."""
        return self.Q.get((s, a), 0.0)

    def best_action(self, s: Any) -> int:
        """Determina la mejor acción (greedy) para un estado dado.

        Args:
            s: Estado actual.

        Returns:
            La acción con el mayor valor Q.
        """
        best_a = self.actions[0]
        best_v = self.get_q(s, best_a)
        for a in self.actions[1:]:
            v = self.get_q(s, a)
            if v > best_v:
                best_v = v
                best_a = a
        return best_a

    def choose_action(self, s: Any, training: bool = True) -> int:
        """Selecciona una acción usando una política epsilon-greedy.

        Args:
            s: Estado actual.
            training: Si es True, explora con probabilidad epsilon.

        Returns:
            La acción seleccionada.
        """
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.best_action(s)

    def update(self, s: Any, a: int, r: float, s2: Any, done: bool) -> None:
        """Actualiza la tabla Q basándose en la experiencia (s, a, r, s').

        Aplica la regla de actualización estándar de Q-Learning:
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s,a))

        Args:
            s: Estado previo.
            a: Acción tomada.
            r: Recompensa recibida.
            s2: Nuevo estado resultante.
            done: Indica si el episodio ha terminado.
        """
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
        """Reduce el valor de epsilon según el factor de decaimiento."""
        self.epsilon = max(self.cfg.eps_min, self.epsilon * self.cfg.eps_decay)

    def save(self, path: str) -> None:
        """Guarda el estado del agente (tabla Q, config) en disco."""
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
        """Carga un agente desde un archivo en disco."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        agent = QLearningAgent(payload["actions"], payload["cfg"])
        agent.Q = payload["Q"]
        agent.epsilon = payload.get("epsilon", payload["cfg"].eps_min)
        return agent
