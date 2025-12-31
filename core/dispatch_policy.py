# core/dispatch_policy.py
"""Política de despacho y configuración de Q-Learning.

Define la política de toma de decisiones (Q-learning tabular), las configuraciones
del agente y las funciones de discretización de estado utilizadas para convertir
el estado continuo de la simulación en estados discretos para el aprendizaje.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import random

State = Tuple[int, int, int, int, int, int, int, int]  # bins
Action = int

A_ASSIGN_URGENT_NEAREST = 0
A_ASSIGN_ANY_NEAREST    = 1
A_WAIT                  = 2
A_REPLAN_TRAFFIC        = 3

ALL_ACTIONS = [A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC]


@dataclass
class QLearningConfig:
    """Configuración de hiperparámetros para Q-Learning.

    Attributes:
        alpha: Tasa de aprendizaje (learning rate).
        gamma: Factor de descuento.
        epsilon: Probabilidad inicial de exploración (epsilon-greedy).
        epsilon_min: Valor mínimo de epsilon.
        epsilon_decay: Factor de decaimiento de epsilon por episodio.
    """
    alpha: float = 0.15
    gamma: float = 0.95
    epsilon: float = 0.20
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995


class DispatchPolicy:
    """Política de Aprendizaje por Refuerzo (Q-learning tabular).

    Decide la estrategia de alto nivel (acción abstracta) a tomar en cada paso.
    La ejecución concreta de la acción (micro-asignación) es delegada al AssignmentEngine.

    Attributes:
        cfg: Configuración de Q-Learning.
        rng: Generador de números aleatorios para reproducibilidad.
        Q: Tabla Q almacena el valor (state, action) -> float.
    """

    def __init__(self, cfg: QLearningConfig, seed: int = 123):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.Q: Dict[Tuple[State, Action], float] = {}

    def get_Q(self, s: State, a: Action) -> float:
        """Obtiene el valor Q para un par estado-acción."""
        return self.Q.get((s, a), 0.0)

    def set_Q(self, s: State, a: Action, v: float) -> None:
        """Actualiza el valor Q para un par estado-acción."""
        self.Q[(s, a)] = v

    def choose_action(self, s: State) -> Action:
        """Selecciona una acción usando una política epsilon-greedy.

        Args:
            s: Estado actual discretizado.

        Returns:
            La acción seleccionada (int).
        """
        # epsilon-greedy
        if self.rng.random() < self.cfg.epsilon:
            return self.rng.choice(ALL_ACTIONS)

        # greedy: elegir la acción con mayor Q
        qs = [(self.get_Q(s, a), a) for a in ALL_ACTIONS]
        qs.sort(reverse=True, key=lambda x: x[0])
        return qs[0][1]

    def update(self, s: State, a: Action, r: float, s2: State, done: bool) -> None:
        """Realiza una actualización de Q-Learning (SARSA/Q-learning update).

        Args:
            s: Estado anterior.
            a: Acción tomada.
            r: Recompensa recibida.
            s2: Nuevo estado resultante.
            done: Indica si el episodio ha terminado.
        """
        old = self.get_Q(s, a)
        if done:
            target = r
        else:
            best_next = max(self.get_Q(s2, a2) for a2 in ALL_ACTIONS)
            target = r + self.cfg.gamma * best_next
        new = old + self.cfg.alpha * (target - old)
        self.set_Q(s, a, new)

    def decay_epsilon(self) -> None:
        """Reduce el valor de epsilon multiplicándolo por el factor de decaimiento."""
        self.cfg.epsilon = max(self.cfg.epsilon_min, self.cfg.epsilon * self.cfg.epsilon_decay)


# -----------------------------
# Discretización del estado
# -----------------------------

def bin_time(t: int, episode_len: int) -> int:
    """Discretiza el tiempo transcurrido en 5 segmentos (0-4)."""
    # 0..4
    frac = t / max(1, episode_len)
    if frac < 0.2: return 0
    if frac < 0.4: return 1
    if frac < 0.6: return 2
    if frac < 0.8: return 3
    return 4

def bin_pending(n: int) -> int:
    """Discretiza la cantidad de pedidos pendientes.
    
    Bins: 0, 1-2, 3-5, 6-10, >10 -> (0-4)
    """
    if n == 0: return 0
    if n <= 2: return 1
    if n <= 5: return 2
    if n <= 10: return 3
    return 4

def bin_ratio_urgent(r: float) -> int:
    """Discretiza el ratio de pedidos urgentes.
    
    Bins: 0, 1-25%, 26-50%, 51-75%, 76-100% -> (0-4)
    """
    if r <= 0.0: return 0
    if r <= 0.25: return 1
    if r <= 0.50: return 2
    if r <= 0.75: return 3
    return 4

def bin_free_riders(n: int) -> int:
    """Discretiza el número de riders libres.
    
    Bins: 0, 1, 2, 3+ -> (0-3)
    """
    if n == 0: return 0
    if n == 1: return 1
    if n == 2: return 2
    return 3  # 3+

def bin_fatigue(avg: float) -> int:
    """Discretiza la fatiga promedio de la flota.
    
    Bins: <1.0, <2.5, >=2.5 -> (0-2)
    """
    # (ajústalo luego). v1: fatiga crece ~0.05 por movimiento
    if avg < 1.0: return 0
    if avg < 2.5: return 1
    return 2

def bin_imbalance(std_deliveries: float) -> int:
    """Discretiza el desbalance de entregas (desviación estándar).
    
    Bins: <0.5, <1.5, >=1.5 -> (0-2)
    """
    if std_deliveries < 0.5: return 0
    if std_deliveries < 1.5: return 1
    return 2

def bin_traffic(level: str) -> int:
    """Convierte el nivel de tráfico a entero.
    
    Map: low->0, medium->1, high->2
    """
    return {"low": 0, "medium": 1, "high": 2}.get(level, 0)

def bin_closures(n: int) -> int:
    """Discretiza el número de cierres de calles.
    
    Bins: 0, 1, 2+ -> (0-2)
    """
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
    """Construye la tupla de estado discretizado a partir de variables crudas.

    Args:
        t: Tiempo actual.
        episode_len: Duración total del episodio.
        pending: Cantidad de pedidos pendientes.
        urgent_ratio: Proporción de pedidos urgentes.
        free_riders: Cantidad de riders libres.
        avg_fatigue: Fatiga promedio.
        std_deliveries: Desviación estándar de entregas por rider.
        traffic_level: Nivel de tráfico global ('low', 'medium', 'high').
        closures: Cantidad de calles cerradas.

    Returns:
        Tupla State con los valores discretizados.
    """
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
