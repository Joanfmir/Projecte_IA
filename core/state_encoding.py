# core/state_encoding.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _bucket(v: float, cuts: Tuple[float, ...]) -> int:
    """
    Devuelve índice de bucket según puntos de corte.
    cuts=(a,b,c) => 0..len(cuts)
    """
    for i, c in enumerate(cuts):
        if v <= c:
            return i
    return len(cuts)


def _traffic_to_int(level: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(level, 0)


@dataclass
class StateEncoder:
    """
    Encoder discreto para Q-learning.
    IMPORTANTÍSIMO: mantener el estado pequeño (pocos buckets) o explota.
    """
    # buckets sencillos
    pending_cuts: Tuple[float, ...] = (2, 5, 9, 14, 20)
    urgent_ratio_cuts: Tuple[float, ...] = (0.0, 0.15, 0.35, 0.6)
    free_cuts: Tuple[float, ...] = (0, 1, 2, 3, 5)
    fat_cuts: Tuple[float, ...] = (0.5, 1.5, 3.0, 5.0, 8.0)
    closures_cuts: Tuple[float, ...] = (0, 1, 3, 6, 10)

    def encode(self, snap: Dict[str, Any]) -> Tuple[int, ...]:
        t = int(snap.get("t", 0))
        episode_len = int(snap.get("episode_len", 1)) if snap.get("episode_len") is not None else 1

        pending = len(snap.get("pending_orders", []))

        # urgent_ratio: si tu pending_orders lleva priority en la tupla (loc, priority, ...)
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

        # tráfico global (si existe)
        traffic_level = snap.get("traffic", "low")
        traffic_global = _traffic_to_int(str(traffic_level))

        # ✅ tráfico por zonas (esto es lo que te faltaba para que “aprenda” con zonas)
        zones = snap.get("traffic_zones", {}) or {}
        z0 = _traffic_to_int(str(zones.get(0, "low")))
        z1 = _traffic_to_int(str(zones.get(1, "low")))
        z2 = _traffic_to_int(str(zones.get(2, "low")))
        z3 = _traffic_to_int(str(zones.get(3, "low")))

        # tiempo normalizado discretizado
        t_norm = t / max(1, episode_len)
        t_b = _bucket(t_norm, (0.2, 0.4, 0.6, 0.8))

        # discretización resto
        pending_b = _bucket(pending, self.pending_cuts)
        urgent_b = _bucket(_clip(urgent_ratio, 0.0, 1.0), self.urgent_ratio_cuts)
        free_b = _bucket(free, self.free_cuts)
        fat_b = _bucket(_clip(avg_fat, 0.0, 50.0), self.fat_cuts)
        clos_b = _bucket(closures, self.closures_cuts)

        # Estado final (ojo: mantenerlo compacto)
        return (
            t_b,
            pending_b,
            urgent_b,
            free_b,
            fat_b,
            clos_b,
            traffic_global,
            z0, z1, z2, z3,  # ✅ añadido
        )
