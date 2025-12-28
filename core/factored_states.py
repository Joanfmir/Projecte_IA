# core/factored_states.py
"""
Codificación de estados para Q-Learning factorizado.
2 tipos de estado para 2 Q-tables: Q1 (asignación) y Q3 (incidente/tráfico).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

Node = Tuple[int, int]


# ─────────────────────────────────────────────────────────────
# Funciones de discretización (binning)
# ─────────────────────────────────────────────────────────────


def bin_time(t: int, episode_len: int) -> int:
    """Progreso del episodio en 5 franjas (0-4)."""
    frac = t / max(1, episode_len)
    if frac < 0.2:
        return 0
    if frac < 0.4:
        return 1
    if frac < 0.6:
        return 2
    if frac < 0.8:
        return 3
    return 4


def bin_pending_unassigned(n: int) -> int:
    """Pedidos sin asignar: 0, 1-2, 3-5, 6-10, 11+ → 0-4."""
    if n == 0:
        return 0
    if n <= 2:
        return 1
    if n <= 5:
        return 2
    if n <= 10:
        return 3
    return 4


def bin_urgent(n: int) -> int:
    """Pedidos urgentes: 0, 1, 2-3, 4+ → 0-3."""
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n <= 3:
        return 2
    return 3


def bin_free_riders(n: int) -> int:
    """Riders elegibles: 0, 1, 2, 3+ → 0-3."""
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    return 3


def bin_min_slack(slack: int) -> int:
    """Tiempo mínimo hasta deadline: ≤0, 1-4, 5-8, 9-15, 16+ → 0-4."""
    if slack <= 0:
        return 0
    if slack <= 4:
        return 1
    if slack <= 8:
        return 2
    if slack <= 15:
        return 3
    return 4


def bin_zones_congested(count: int) -> int:
    """Zonas con tráfico medium/high: 0, 1, 2, 3-4 → 0-3."""
    if count == 0:
        return 0
    if count == 1:
        return 1
    if count == 2:
        return 2
    return 3  # 3-4 zonas congestionadas


def bin_backlog(n: int) -> int:
    """Pedidos pendientes totales: 0, 1-3, 4-7, 8-15, 16+ → 0-4."""
    if n == 0:
        return 0
    if n <= 3:
        return 1
    if n <= 7:
        return 2
    if n <= 15:
        return 3
    return 4


def bin_urgent_ratio(ratio: float) -> int:
    """Ratio urgentes/total: 0%, 1-25%, 26-50%, 51-75%, 76-100% → 0-4."""
    if ratio <= 0:
        return 0
    if ratio <= 0.25:
        return 1
    if ratio <= 0.50:
        return 2
    if ratio <= 0.75:
        return 3
    return 4


def bin_imbalance(std: float) -> int:
    """Desbalance de carga (std de pedidos asignados): <0.5, <1.5, ≥1.5 → 0-2."""
    if std < 0.5:
        return 0
    if std < 1.5:
        return 1
    return 2


def bin_fatigue(avg: float) -> int:
    """Fatiga promedio: <1.0, <2.5, ≥2.5 → 0-2."""
    if avg < 1.0:
        return 0
    if avg < 2.5:
        return 1
    return 2


def bin_delta_traffic(delta: float) -> int:
    """Cambio en presión de tráfico normalizada: 0, pequeño, medio, grande → 0-3."""
    if delta <= 0.0:
        return 0
    if delta <= 0.3:
        return 1
    if delta <= 0.8:
        return 2
    return 3


def bin_busy_riders(n: int) -> int:
    """Riders en ruta (ocupados): 0, 1, 2-3, 4+ → 0-3."""
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n <= 3:
        return 2
    return 3


def bin_riders_at_restaurant(n: int) -> int:
    """Riders en tienda (listos para salir): 0, 1, 2+ → 0-2."""
    if n == 0:
        return 0
    if n == 1:
        return 1
    return 2


def bin_min_rider_distance(dist: float) -> int:
    """Distancia mínima rider elegible -> pedido más cercano: 0-3, 4-8, 9-15, 16+ → 0-3."""
    if dist <= 3:
        return 0
    if dist <= 8:
        return 1
    if dist <= 15:
        return 2
    return 3


# ─────────────────────────────────────────────────────────────
# Extracción de features desde snapshot
# ─────────────────────────────────────────────────────────────


def traffic_pressure_from_zones(traffic_zones: Dict) -> float:
    """Calcula presión de tráfico normalizada (promedio) desde zonas."""
    mapping = {"low": 1.0, "medium": 1.5, "high": 2.2}
    if not traffic_zones:
        return 1.0
    vals = [mapping.get(lvl, 1.0) for lvl in traffic_zones.values()]
    return sum(vals) / len(vals)


def extract_features(
    snap: Dict, episode_len: int, prev_traffic_pressure: float = 1.0
) -> Dict:
    """
    Extrae todos los features necesarios desde un snapshot del simulador.
    Retorna un diccionario con valores crudos (sin discretizar).
    """
    t = snap.get("t", 0)
    restaurant = snap.get("restaurant", (0, 0))

    # Pedidos
    pending_orders = snap.get("pending_orders", [])
    # pending_orders es lista de (dropoff, priority, deadline, assigned_to)

    unassigned = [o for o in pending_orders if o[3] is None]
    pending_total = len(pending_orders)
    pending_unassigned = len(unassigned)

    # Urgentes (priority > 1 O deadline - t <= 8)
    urgent_unassigned = sum(
        1
        for (_, priority, deadline, assigned) in unassigned
        if priority > 1 or (deadline - t) <= 8
    )

    urgent_total = sum(
        1
        for (_, priority, deadline, _) in pending_orders
        if priority > 1 or (deadline - t) <= 8
    )

    # Min slack (tiempo mínimo hasta deadline de pedidos sin asignar)
    if unassigned:
        min_slack = min((deadline - t) for (_, _, deadline, _) in unassigned)
    else:
        min_slack = 999  # No hay pedidos sin asignar

    # Riders
    riders = snap.get("riders", [])
    restaurant = snap.get("restaurant", (0, 0))

    # F3 FIX: Elegibles ALINEADO con AssignmentEngine._eligible_riders()
    # Criterios: can_take_more, NOT resting, (available OR at_restaurant)
    def is_eligible(r):
        has_capacity = len(r.get("assigned", [])) < 2
        resting = r.get("resting", False)
        available = r.get("available", False)
        at_restaurant = tuple(r.get("pos", (0, 0))) == tuple(restaurant)
        return has_capacity and (not resting) and (available or at_restaurant)

    free_riders = sum(1 for r in riders if is_eligible(r))

    # En restaurante
    riders_at_restaurant = sum(
        1 for r in riders if tuple(r.get("pos", (0, 0))) == tuple(restaurant)
    )

    # Busy (en ruta, no available)
    busy_riders = sum(
        1 for r in riders if not r.get("available", True) or len(r.get("route", [])) > 0
    )

    # Fatiga promedio
    fatigues = [r.get("fatigue", 0) for r in riders]
    avg_fatigue = sum(fatigues) / len(fatigues) if fatigues else 0

    # Imbalance (std de pedidos asignados)
    assigned_counts = [len(r.get("assigned", [])) for r in riders]
    if len(assigned_counts) >= 2:
        mean_assigned = sum(assigned_counts) / len(assigned_counts)
        variance = sum((x - mean_assigned) ** 2 for x in assigned_counts) / len(
            assigned_counts
        )
        std_assigned = variance**0.5
    else:
        std_assigned = 0

    # NUEVO: Distancia mínima rider elegible -> pedido sin asignar (octile)
    min_rider_to_order = 999.0
    restaurant = snap.get("restaurant", (0, 0))
    for o_data in unassigned:
        # o_data = (dropoff, priority, deadline, assigned_to)
        ox, oy = o_data[0]  # dropoff position
        for r in riders:
            if is_eligible(r):
                rx, ry = r.get("pos", (0, 0))
                # Distancia octile: rider -> restaurant -> dropoff
                # Si rider está en restaurante, solo distancia a dropoff
                if (rx, ry) == tuple(restaurant):
                    dx, dy = abs(rx - ox), abs(ry - oy)
                else:
                    # rider -> restaurant + restaurant -> dropoff
                    d1x, d1y = abs(rx - restaurant[0]), abs(ry - restaurant[1])
                    d2x, d2y = abs(restaurant[0] - ox), abs(restaurant[1] - oy)
                    dx = d1x + d2x
                    dy = d1y + d2y
                octile = max(dx, dy) + 0.414 * min(dx, dy)
                min_rider_to_order = min(min_rider_to_order, octile)

    # Tráfico por zonas - NORMALIZADO
    traffic_zones = snap.get("traffic_zones", {})
    traffic_global = snap.get("traffic", "low")

    # Contar zonas congestionadas (medium o high)
    if traffic_zones:
        zones_congested = sum(
            1 for lvl in traffic_zones.values() if lvl in ("medium", "high")
        )
        # Presión normalizada (promedio)
        current_pressure = traffic_pressure_from_zones(traffic_zones)
    else:
        # Sin zonas, usar tráfico global
        zones_congested = 1 if traffic_global in ("medium", "high") else 0
        mapping = {"low": 1.0, "medium": 1.5, "high": 2.2}
        current_pressure = mapping.get(traffic_global, 1.0)

    delta_traffic = abs(current_pressure - prev_traffic_pressure)

    # Urgent ratio
    urgent_ratio = urgent_total / pending_total if pending_total > 0 else 0

    return {
        "t": t,
        "episode_len": episode_len,
        "pending_total": pending_total,
        "pending_unassigned": pending_unassigned,
        "urgent_unassigned": urgent_unassigned,
        "urgent_total": urgent_total,
        "urgent_ratio": urgent_ratio,
        "min_slack": min_slack,
        "free_riders": free_riders,
        "riders_at_restaurant": riders_at_restaurant,
        "busy_riders": busy_riders,
        "avg_fatigue": avg_fatigue,
        "std_assigned": std_assigned,
        "zones_congested": zones_congested,
        "traffic_pressure": current_pressure,
        "delta_traffic": delta_traffic,
        "min_rider_to_order": min_rider_to_order,  # NUEVO
    }


# ─────────────────────────────────────────────────────────────
# Encoder de Estados Factorizados
# ─────────────────────────────────────────────────────────────


@dataclass
class FactoredStateEncoder:
    """
    Genera estados discretizados para 2 Q-tables: Q1 (asignación) y Q3 (incidente).
    IMPORTANTE: Usar update_prev=False en encode_all() durante choose_action y update.
    Llamar commit() exactamente una vez por tick para actualizar prev_traffic_pressure.
    """

    episode_len: int = 900
    prev_traffic_pressure: float = 1.0  # Normalizado

    # Umbral para activar Q3 (cambio significativo de tráfico)
    delta_traffic_threshold: float = 0.1

    def reset(self) -> None:
        """Resetea estado interno. Llamar al inicio de cada episodio."""
        self.prev_traffic_pressure = 1.0

    def commit(self, snap: Dict) -> None:
        """
        Actualiza prev_traffic_pressure. Llamar EXACTAMENTE una vez por tick,
        después de que se haya procesado el update de Q-learning.
        """
        traffic_zones = snap.get("traffic_zones", {})
        if traffic_zones:
            self.prev_traffic_pressure = traffic_pressure_from_zones(traffic_zones)
        else:
            traffic_global = snap.get("traffic", "low")
            mapping = {"low": 1.0, "medium": 1.5, "high": 2.2}
            self.prev_traffic_pressure = mapping.get(traffic_global, 1.0)

    def encode_all(self, snap: Dict, update_prev: bool = False) -> Dict[str, Tuple]:
        """
        Retorna los 2 estados discretizados (Q1 y Q3).

        Args:
            snap: Snapshot del simulador
            update_prev: DEPRECATED, usar commit() en su lugar.

        Returns:
            {
                "s_assign": tuple de 8 ints para Q₁ (añadido distancia),
                "s_incident": tuple de 4 ints para Q₃,
                "features": dict con valores crudos (para debug/logging)
            }
        """
        features = extract_features(snap, self.episode_len, self.prev_traffic_pressure)

        # DEPRECATED: preferir commit() explícito
        if update_prev:
            self.prev_traffic_pressure = features["traffic_pressure"]

        # Q₁: Estado de Asignación (8 dims - AÑADIDO distancia)
        s_assign = (
            bin_time(features["t"], features["episode_len"]),
            bin_pending_unassigned(features["pending_unassigned"]),
            bin_urgent(features["urgent_unassigned"]),
            bin_free_riders(features["free_riders"]),
            bin_min_slack(features["min_slack"]),
            bin_zones_congested(features["zones_congested"]),
            bin_riders_at_restaurant(features["riders_at_restaurant"]),
            bin_min_rider_distance(features["min_rider_to_order"]),  # NUEVO
        )

        # Q₃: Estado de Incidente (4 dims)
        s_incident = (
            bin_delta_traffic(features["delta_traffic"]),
            bin_busy_riders(features["busy_riders"]),
            bin_urgent(features["urgent_unassigned"]),
            bin_backlog(features["pending_total"]),
        )

        return {
            "s_assign": s_assign,
            "s_incident": s_incident,
            "features": features,
        }

    def should_use_q1(self, features: Dict) -> bool:
        """¿Hay trabajo por asignar?"""
        return features["pending_unassigned"] > 0 and features["free_riders"] > 0

    def should_use_q3(self, features: Dict) -> bool:
        """
        ¿Hubo cambio significativo de tráfico?
        Activación por SEÑAL (delta_traffic > umbral), no por reloj.
        """
        return features["delta_traffic"] >= self.delta_traffic_threshold


# ─────────────────────────────────────────────────────────────
# Cálculo de tamaño del espacio de estados
# ─────────────────────────────────────────────────────────────


def state_space_sizes() -> Dict[str, int]:
    """Retorna el tamaño de cada espacio de estados (Q1 y Q3 solamente)."""
    # Q₁: 5 * 5 * 4 * 4 * 5 * 4 * 3 * 4 = 96,000 (añadido bin_min_rider_distance x4)
    q1_size = 5 * 5 * 4 * 4 * 5 * 4 * 3 * 4

    # Q₃: 4 * 4 * 4 * 5 = 320
    q3_size = 4 * 4 * 4 * 5

    return {
        "Q1_assign": q1_size,
        "Q3_incident": q3_size,
        "total": q1_size + q3_size,
    }
