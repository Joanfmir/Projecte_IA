# core/factored_states.py
"""Codificación de estados para Q-Learning factorizado.

Define el esquema de discretización (binning) y codificación de estados necesario
para las dos tablas Q (Q1: Asignación, Q3: Incidente). Transforma los datos crudos
de la simulación en tuplas de enteros manejables por el algoritmo de aprendizaje.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from core.route_planner import RoutePlanner

Node = Tuple[int, int]


# ─────────────────────────────────────────────────────────────
# Funciones de discretización (binning)
# ─────────────────────────────────────────────────────────────


def bin_time(t: int, episode_len: int) -> int:
    """Discretiza el progreso del episodio en 5 tramos.
    
    Returns:
        Entero 0-4 indicando el progreso (0-20%, 20-40%, etc.).
    """
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
    """Discretiza la cantidad de pedidos sin asignar (0-4)."""
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
    """Discretiza la cantidad de pedidos urgentes (0-3)."""
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n <= 3:
        return 2
    return 3


def bin_free_riders(n: int) -> int:
    """Discretiza la cantidad de riders elegibles (0-3)."""
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    return 3


def bin_min_slack(slack: int) -> int:
    """Discretiza el tiempo mínimo (slack) hasta el deadline (0-4)."""
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
    """Discretiza min_slack incluyendo un valor centinela para 'sin pedidos'.

    Returns:
        0-4 para valores normales de slack.
        5 si no hay pedidos (slack >= 300).
    """
    if slack >= 300:
        return 5  # sentinel alto (sin pedidos sin asignar)
    if slack == 0:
        return 0
    if slack <= 4:
        return 1
    if slack <= 8:
        return 2
    if slack <= 15:
        return 3
    return 4


def bin_zones_congested(count: int) -> int:
    """Discretiza la cantidad de zonas con tráfico medio/alto (0-3)."""
    if count == 0:
        return 0
    if count == 1:
        return 1
    if count == 2:
        return 2
    return 3  # 3-4 zonas congestionadas


def bin_backlog(n: int) -> int:
    """Discretiza el total de pedidos pendientes (backlog) (0-4)."""
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
    """Discretiza la proporción de pedidos urgentes sobre el total (0-4)."""
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
    """Discretiza el desbalance de carga entre riders (0-2)."""
    if std < 0.5:
        return 0
    if std < 1.5:
        return 1
    return 2


def bin_fatigue(avg: float) -> int:
    """Discretiza la fatiga promedio de la flota (0-2)."""
    if avg < 1.0:
        return 0
    if avg < 2.5:
        return 1
    return 2


def bin_delta_traffic(delta: float) -> int:
    """Discretiza el cambio en la presión de tráfico (0-3)."""
    if delta <= 0.0:
        return 0
    if delta <= 0.3:
        return 1
    if delta <= 0.8:
        return 2
    return 3


def bin_busy_riders(n: int) -> int:
    """Discretiza la cantidad de riders ocupados/en ruta (0-3)."""
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n <= 3:
        return 2
    return 3


def bin_riders_at_restaurant(n: int) -> int:
    """Discretiza la cantidad de riders esperando en el restaurante (0-2)."""
    if n == 0:
        return 0
    if n == 1:
        return 1
    return 2


def bin_capacity_mix(partial: int, full: int) -> int:
    """Clasifica el estado de capacidad de la flota.

    Returns:
        0: Todos vacíos.
        1: Hay parciales (con espacio), pero ninguno lleno.
        2: Hay riders llenos (3+ activos).
    """
    if full > 0:
        return 2
    if partial > 0:
        return 1
    return 0


def bin_min_rider_distance(dist: float) -> int:
    """Discretiza la distancia mínima de un rider a un pedido (0-3)."""
    if dist <= 3:
        return 0
    if dist <= 8:
        return 1
    if dist <= 15:
        return 2
    return 3


def bin_distance_with_sentinel(dist: float) -> int:
    """Discretiza distancia/ETA incluyendo valor centinela (<0).

    Returns:
        0 si dist < 0 (valor inválido/centinela).
        1-4 para distancias válidas.
    """
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
    """Calcula la presión promedio de tráfico normalizada basada en zonas."""
    mapping = {"low": 1.0, "medium": 1.5, "high": 2.2}
    if not traffic_zones:
        return 1.0
    vals = [mapping.get(lvl, 1.0) for lvl in traffic_zones.values()]
    return sum(vals) / len(vals)


def extract_features(
    snap: Dict, episode_len: int, prev_traffic_pressure: float = 1.0
) -> Dict:
    """Extrae características crudas del estado de la simulación.

    Procesa el snapshot para obtener métricas sobre pedidos, riders y tráfico
    antes de la discretización.

    Args:
        snap: Snapshot del simulador.
        episode_len: Duración total del episodio.
        prev_traffic_pressure: Presión de tráfico del tick anterior.

    Returns:
        Diccionario con features calculados (pendientes, urgentes, distancias, etc.).
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
            oid = item[4] if len(item) > 4 else idx
            pending_details.append(
                {
                    "id": oid,
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
        min_slack = 999  # Sentinel alto: no hay pedidos sin asignar

    # Riders
    riders = snap.get("riders", [])
    restaurant = snap.get("restaurant", (0, 0))

    def active_count(r: Dict) -> int:
        return len(r.get("assigned", []))

    # F3 FIX: Elegibles ALINEADO con AssignmentEngine._eligible_riders()
    # Criterios: can_take_more, NOT resting, (available OR at_restaurant)
    def is_eligible(r):
        assigned_len = active_count(r)
        capacity = r.get("capacity", 3)
        has_capacity = assigned_len < capacity
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
        capacity_r = r.get("capacity", 3)
        resting = r.get("resting", False)
        if count == 0:
            empty_riders += 1
        elif count < capacity_r:
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
                    dist = RoutePlanner.heuristic((rx, ry), (ox, oy))
                else:
                    dist = RoutePlanner.heuristic((rx, ry), tuple(restaurant)) + RoutePlanner.heuristic(
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
        drop = candidate_order["dropoff"]
        for r in partial_pool:
            rx, ry = r.get("pos", (0, 0))
            if (rx, ry) != tuple(restaurant):
                continue
            eta = RoutePlanner.heuristic(tuple(restaurant), drop)
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
        "capacity_mix": bin_capacity_mix(partial_riders, full_riders),
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
    """Codificador de estados para el agente factorizado.

    Convierte snapshots de simulación en tuplas de estado discretizadas para:
    - Q1: Decisiones de asignación.
    - Q3: Decisiones de replanificación por incidentes.

    Mantiene estado interno (presión de tráfico previa) para detectar cambios.

    Attributes:
        episode_len: Duración máxima del episodio para normalizar tiempo.
        prev_traffic_pressure: Presión de tráfico del paso anterior (para calcular delta).
        delta_traffic_threshold: Umbral de cambio de tráfico para activar Q3.
    """

    episode_len: int = 900
    prev_traffic_pressure: float = 1.0  # Normalizado

    # Umbral para activar Q3 (cambio significativo de tráfico)
    delta_traffic_threshold: float = 0.1

    def reset(self) -> None:
        """Resetea el estado interno al inicio de un nuevo episodio."""
        self.prev_traffic_pressure = 1.0

    def commit(self, snap: Dict) -> None:
        """Actualiza la presión de tráfico histórica.

        Debe llamarse una vez por tick, después de procesar actualizaciones Q.

        Args:
            snap: Snapshot actual.
        """
        traffic_zones = snap.get("traffic_zones", {})
        if traffic_zones:
            self.prev_traffic_pressure = traffic_pressure_from_zones(traffic_zones)
        else:
            traffic_global = snap.get("traffic", "low")
            mapping = {"low": 1.0, "medium": 1.5, "high": 2.2}
            self.prev_traffic_pressure = mapping.get(traffic_global, 1.0)

    def encode_all(self, snap: Dict, update_prev: bool = False) -> Dict[str, Tuple]:
        """Genera los estados discretizados para Q1 y Q3.

        Args:
            snap: Snapshot del simulador.
            update_prev: DEPRECATED. Usar commit() en su lugar.

        Returns:
            Diccionario con las claves:
            - 's_assign': Tuple de estado para Q1.
            - 's_incident': Tuple de estado para Q3.
            - 'features': Diccionario de features crudos.
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
            bin_riders_at_restaurant(features["riders_at_restaurant"]),
            bin_distance_with_sentinel(features["min_rider_to_order"]),
            features["capacity_mix"],
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
        """Determina si se debe consultar Q1 (hay trabajo por asignar)."""
        return features["pending_unassigned"] > 0 and features["free_riders"] > 0

    def should_use_q3(self, features: Dict) -> bool:
        """Determina si se debe consultar Q3 (cambio significativo de tráfico)."""
        return features["delta_traffic"] >= self.delta_traffic_threshold


# ─────────────────────────────────────────────────────────────
# Cálculo de tamaño del espacio de estados
# ─────────────────────────────────────────────────────────────


def state_space_sizes() -> Dict[str, int]:
    """Calcula y retorna el tamaño combinatorio de los espacios de estados.

    Returns:
        Diccionario con tamaños de 'Q1_assign', 'Q3_incident' y 'total'.
    """
    # Q₁: 5 * 5 * 4 * 4 * 6 * 3 * 5 * 3 * 5 = 540,000
    q1_size = 5 * 5 * 4 * 4 * 6 * 3 * 5 * 3 * 5

    # Q₃: 4 * 4 * 4 * 5 = 320
    q3_size = 4 * 4 * 4 * 5

    return {
        "Q1_assign": q1_size,
        "Q3_incident": q3_size,
        "total": q1_size + q3_size,
    }
