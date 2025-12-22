# core/state_encoding.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


def bin_by_thresholds(x: int, thresholds: Tuple[int, ...]) -> int:
    """
    Devuelve el índice del bin según thresholds.
    Ej: thresholds=(0,3,7) -> bins:
      x<=0 ->0
      1..3 ->1
      4..7 ->2
      >=8 ->3
    """
    if x <= thresholds[0]:
        return 0
    for i in range(1, len(thresholds)):
        if x <= thresholds[i]:
            return i
    return len(thresholds)


@dataclass(frozen=True)
class StateEncoder:
    """
    Convierte snapshot -> estado DISCRETO (hashable) para Q-learning.
    Usamos sólo info del snapshot para no depender de estructuras internas.
    """

    pending_thresholds: Tuple[int, ...] = (0, 3, 7)
    urgent_thresholds: Tuple[int, ...] = (0, 2, 4)
    free_thresholds: Tuple[int, ...] = (0, 1, 2)

    closure_thresholds: Tuple[int, ...] = (0, 2, 5)

    def encode(self, snap: Dict) -> Tuple:
        pending = len(snap["pending_orders"])

        # urgent: priority>1
        urgent = 0
        for (loc, priority, deadline, assigned_to) in snap["pending_orders"]:
            if priority > 1:
                urgent += 1

        # free riders (aprox): si no lleva pedido y no tiene ruta -> libre
        free = 0
        for r in snap["riders"]:
            carrying = r.get("carrying", None)
            route = r.get("route", [])
            if carrying is None and (route is None or len(route) == 0):
                free += 1

        traffic = snap.get("traffic", "low")
        traffic_id = {"low": 0, "medium": 1, "high": 2}.get(traffic, 0)

        closures = int(snap.get("closures", 0))

        s = (
            bin_by_thresholds(pending, self.pending_thresholds),
            bin_by_thresholds(urgent, self.urgent_thresholds),
            bin_by_thresholds(free, self.free_thresholds),
            traffic_id,
            bin_by_thresholds(closures, self.closure_thresholds),
        )
        return s
