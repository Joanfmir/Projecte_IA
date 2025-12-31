# core/state_encoding.py
"""Codificación de estados (Simple/Baseline).

Este módulo implementa un encoder discreto sencillo para Q-Learning, utilizando
técnicas de bucketing (discretización) para reducir el espacio de estados.
Ideal para baselines o agentes iniciales.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple


def _clip(v: float, lo: float, hi: float) -> float:
    """Restringe un valor dentro de un rango [lo, hi]."""
    return lo if v < lo else hi if v > hi else v


def _bucket(v: float, cuts: Tuple[float, ...]) -> int:
    """Devuelve el índice del bucket correspondiente según puntos de corte.

    Args:
        v: Valor numérico a discretizar.
        cuts: Tupla con los límites superiores de cada bucket.

    Returns:
        Índice del bucket (0 a len(cuts)).
    """
    for i, c in enumerate(cuts):
        if v <= c:
            return i
    return len(cuts)


def _traffic_to_int(level: str) -> int:
    """Convierte el nivel de tráfico (str) a entero (0, 1, 2)."""
    return {"low": 0, "medium": 1, "high": 2}.get(level, 0)


@dataclass
class StateEncoder:
    """Encoder discreto para Q-learning.

    Transforma el diccionario de estado (snapshot) en una tupla de enteros,
    discretizando variables continuas como tiempo, pendientes y fatiga.

    Attributes:
        pending_cuts: Cortes para cantidad de pedidos pendientes.
        urgent_ratio_cuts: Cortes para ratio de urgencia.
        free_cuts: Cortes para cantidad de riders libres.
        fat_cuts: Cortes para fatiga promedio.
        closures_cuts: Cortes para número de calles cerradas.
    """
    # buckets sencillos
    pending_cuts: Tuple[float, ...] = (2, 5, 9, 14, 20)
    urgent_ratio_cuts: Tuple[float, ...] = (0.0, 0.15, 0.35, 0.6)
    free_cuts: Tuple[float, ...] = (0, 1, 2, 3, 5)
    fat_cuts: Tuple[float, ...] = (0.5, 1.5, 3.0, 5.0, 8.0)
    closures_cuts: Tuple[float, ...] = (0, 1, 3, 6, 10)

    def encode(self, snap: Dict[str, Any]) -> Tuple[int, ...]:
        """Genera la representación de estado discreta.

        Args:
            snap: Snapshot del estado de la simulación.

        Returns:
            Tupla de enteros representando el estado.
        """
        t = int(snap.get("t", 0))
        episode_len = int(snap.get("episode_len", 1)) if snap.get("episode_len") is not None else 1

        pending = len(snap.get("pending_orders", []))

        # Calcular urgent_ratio
        pend_list = snap.get("pending_orders", [])
        urgent = 0
        for item in pend_list:
            # item = (loc, priority, deadline, assigned_to)
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pr = item[1]
                if pr is not None and pr > 1:
                    urgent += 1
        urgent_ratio = (urgent / pending) if pending > 0 else 0.0

        riders = snap.get("riders", [])
        free = 0
        fat_sum = 0.0
        for r in riders:
            if r.get("available", True):
                free += 1
            fat_sum += float(r.get("fatigue", 0.0))
        avg_fat = fat_sum / max(1, len(riders))

        closures = int(snap.get("closures", 0))

        # Tráfico global (si existe)
        traffic_level = snap.get("traffic", "low")
        traffic_global = _traffic_to_int(str(traffic_level))

        # Tráfico por zonas
        zones = snap.get("traffic_zones", {}) or {}
        z0 = _traffic_to_int(str(zones.get(0, "low")))
        z1 = _traffic_to_int(str(zones.get(1, "low")))
        z2 = _traffic_to_int(str(zones.get(2, "low")))
        z3 = _traffic_to_int(str(zones.get(3, "low")))

        # Tiempo normalizado discretizado
        t_norm = t / max(1, episode_len)
        t_b = _bucket(t_norm, (0.2, 0.4, 0.6, 0.8))

        # Discretización del resto de variables
        pending_b = _bucket(pending, self.pending_cuts)
        urgent_b = _bucket(_clip(urgent_ratio, 0.0, 1.0), self.urgent_ratio_cuts)
        free_b = _bucket(free, self.free_cuts)
        fat_b = _bucket(_clip(avg_fat, 0.0, 50.0), self.fat_cuts)
        clos_b = _bucket(closures, self.closures_cuts)

        # Estado final
        return (
            t_b,
            pending_b,
            urgent_b,
            free_b,
            fat_b,
            clos_b,
            traffic_global,
            z0, z1, z2, z3,
        )
