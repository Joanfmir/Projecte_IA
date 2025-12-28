# core/factored_states.py
"""
Codificación de estados para Q-Learning factorizado.
2 tipos de estado para 2 Q-tables: Q1 (asignación) y Q3 (incidente/tráfico).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

Node = Tuple[int, int]


def octile(a: Node, b: Node) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + 0.414 * min(dx, dy)


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


def bin_min_slack_with_sentinel(slack: int) -> int:
    """Min slack con bin reservado para 'sin pedidos' (valor <0)."""
    if slack < 0:
        return 0
    if slack <= 0:
        return 1
    if slack <= 4:
        return 2
    if slack <= 8:
        return 3
    if slack <= 15:
        return 4
    return 5


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


def bin_capacity_count(n: int) -> int:
    """Conteo de riders por estado de carga: 0, 1, 2, 3+ → 0-3."""
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    return 3


def bin_min_rider_distance(dist: float) -> int:
    """Distancia mínima rider elegible -> pedido más cercano: 0-3, 4-8, 9-15, 16+ → 0-3."""
    if dist <= 3:
        return 0
    if dist <= 8:
        return 1
    if dist <= 15:
        return 2
    return 3


def bin_distance_with_sentinel(dist: float) -> int:
    """Distancia/ETA con bin reservado para sentinel (<0)."""
    if dist < 0:
        return 0
    if dist <= 3:
        return 1
    if dist <= 8:
        return 2
    if dist <= 15:
        return 3
    return 4


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
    orders_full = snap.get("orders_full", [])
    pending_details: List[Dict] = []
    if orders_full:
        for od in orders_full:
            if od.get("delivered_at") is not None:
                continue
            pending_details.append(
                {
                    "id": od.get("id"),
                    "dropoff": tuple(od.get("dropoff", (0, 0))),
                    "priority": od.get("priority", 1),
                    "deadline": od.get("deadline", 0),
                    "assigned_to": od.get("assigned_to"),
                }
            )
    else:
        pending_orders = snap.get("pending_orders", [])
        for idx, item in enumerate(pending_orders):
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            dropoff, priority, deadline, assigned = item[:4]
            pending_details.append(
                {
                    "id": idx,
                    "dropoff": tuple(dropoff),
                    "priority": priority,
                    "deadline": deadline,
                    "assigned_to": assigned,
                }
            )

    unassigned = [o for o in pending_details if o["assigned_to"] is None]
    pending_total = len(pending_details)
    pending_unassigned = len(unassigned)

    # Urgentes (priority > 1 O deadline - t <= 8)
    urgent_unassigned = sum(
        1
        for o in unassigned
        if o["priority"] > 1 or (o["deadline"] - t) <= 8
    )

    urgent_total = sum(
        1
        for o in pending_details
        if o["priority"] > 1 or (o["deadline"] - t) <= 8
    )

    # Min slack (tiempo mínimo hasta deadline de pedidos sin asignar)
    if unassigned:
        min_slack = min((o["deadline"] - t) for o in unassigned)
    else:
        min_slack = -1  # Sentinel: no hay pedidos sin asignar

    # Riders
    riders = snap.get("riders", [])
    restaurant = snap.get("restaurant", (0, 0))

    def active_count(r: Dict) -> int:
        return len(r.get("assigned", []))

    # F3 FIX: Elegibles ALINEADO con AssignmentEngine._eligible_riders()
    # Criterios: can_take_more, NOT resting, (available OR at_restaurant)
    def is_eligible(r):
        assigned_len = active_count(r)
        has_capacity = assigned_len < 3
        resting = r.get("resting", False)
        available = r.get("available", False)
        at_restaurant = tuple(r.get("pos", (0, 0))) == tuple(restaurant)
        return has_capacity and (not resting) and (available or at_restaurant)

    free_riders = 0
    empty_riders = 0
    partial_riders = 0  # 1 o 2 pedidos activos
    full_riders = 0  # 3 pedidos activos
    partial_pool: List[Dict] = []

    for r in riders:
        count = active_count(r)
        resting = r.get("resting", False)
        if count == 0:
            empty_riders += 1
        elif count < 3:
            partial_riders += 1
            if not resting:
                partial_pool.append(r)
        else:
            full_riders += 1

        if is_eligible(r):
            free_riders += 1

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
    min_rider_to_order = -1.0
    for o_data in unassigned:
        ox, oy = o_data["dropoff"]
        for r in riders:
            if is_eligible(r):
                rx, ry = r.get("pos", (0, 0))
                # Distancia octile: rider -> restaurant -> dropoff
                if (rx, ry) == tuple(restaurant):
                    dist = octile((rx, ry), (ox, oy))
                else:
                    dist = octile((rx, ry), tuple(restaurant)) + octile(
                        tuple(restaurant), (ox, oy)
                    )
                min_rider_to_order = dist if min_rider_to_order < 0 else min(
                    min_rider_to_order, dist
                )

    # Pedido candidato (mínimo slack, tie-break por id)
    candidate_order: Optional[Dict] = None
    if unassigned:
        candidate_order = min(
            unassigned, key=lambda o: (o["deadline"] - t, o["id"])
        )
    closest_partial_eta = -1.0
    if candidate_order and partial_pool:
        for r in partial_pool:
            rx, ry = r.get("pos", (0, 0))
            drop = candidate_order["dropoff"]
            if (rx, ry) == tuple(restaurant):
                eta = octile(tuple(restaurant), drop)
            else:
                eta = octile((rx, ry), tuple(restaurant)) + octile(
                    tuple(restaurant), drop
                )
            closest_partial_eta = eta if closest_partial_eta < 0 else min(
                closest_partial_eta, eta
            )

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
        "candidate_slack": (candidate_order["deadline"] - t)
        if candidate_order
        else -1,
        "free_riders": free_riders,
        "empty_riders": empty_riders,
        "partial_riders": partial_riders,
        "full_riders": full_riders,
        "riders_at_restaurant": riders_at_restaurant,
        "busy_riders": busy_riders,
        "avg_fatigue": avg_fatigue,
        "std_assigned": std_assigned,
        "zones_congested": zones_congested,
        "traffic_pressure": current_pressure,
        "delta_traffic": delta_traffic,
        "min_rider_to_order": min_rider_to_order,  # NUEVO
        "closest_partial_eta": closest_partial_eta,
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

        # Q₁: Estado de Asignación (capacidad granular + batching)
        s_assign = (
            bin_time(features["t"], features["episode_len"]),
            bin_pending_unassigned(features["pending_unassigned"]),
            bin_urgent(features["urgent_unassigned"]),
            bin_free_riders(features["free_riders"]),
            bin_min_slack_with_sentinel(features["min_slack"]),
            bin_zones_congested(features["zones_congested"]),
            bin_riders_at_restaurant(features["riders_at_restaurant"]),
            bin_distance_with_sentinel(features["min_rider_to_order"]),
            bin_capacity_count(features["empty_riders"]),
            bin_capacity_count(features["partial_riders"]),
            bin_capacity_count(features["full_riders"]),
            bin_distance_with_sentinel(features["closest_partial_eta"]),
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
    # Q₁: 5 * 5 * 4 * 4 * 6 * 4 * 3 * 5 * 4 * 4 * 4 * 5 = 46,080,000
    q1_size = 5 * 5 * 4 * 4 * 6 * 4 * 3 * 5 * 4 * 4 * 4 * 5

    # Q₃: 4 * 4 * 4 * 5 = 320
    q3_size = 4 * 4 * 4 * 5

    return {
        "Q1_assign": q1_size,
        "Q3_incident": q3_size,
        "total": q1_size + q3_size,
    }
