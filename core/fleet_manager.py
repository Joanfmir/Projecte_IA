# core/fleet_manager.py
"""Gestión de la flota de repartidores.

Este módulo define la clase Rider y el FleetManager, encargados de rastrear
el estado, ubicación y asignaciones de los repartidores en la simulación.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Node = Tuple[int, int]


@dataclass
class Rider:
    """Representa un repartidor en la simulación.

    Mantiene su estado actual, ubicación, ruta y métricas de rendimiento.

    Attributes:
        rider_id: Identificador único del rider.
        position: Ubicación actual (x, y).
        speed: Velocidad de movimiento (futuro uso).
        available: Si está disponible para recibir nuevas asignaciones.
        fatigue: Nivel de fatiga acumulado.
        distance_travelled: Distancia total recorrida.
        deliveries_done: Cantidad de entregas completadas.
        capacity: Capacidad máxima de pedidos simultáneos.
        assigned_order_ids: IDs de pedidos asignados actualmente.
        waypoints: Lista de puntos de paso (destino intermedio).
        route: Lista de nodos que forman la ruta actual paso a paso.
        has_picked_up: Indica si ya ha recogido los pedidos (si aplica al flujo actual).
        delivery_queue: Orden de IDs de pedidos a entregar.
        resting: Si está en descanso forzado por fatiga.
        wait_until: Tick de simulación hasta el cual debe esperar (batching en restaurante).
    """
    rider_id: int
    position: Node
    speed: float = 1.0  # (por ahora no lo usamos como “pasos por tick”, pero lo dejamos)
    available: bool = True

    # --- Métricas ---
    fatigue: float = 0.0
    distance_travelled: float = 0.0
    deliveries_done: int = 0

    # --- Capacidad máxima de pedidos activos ---
    capacity: int = 3
    assigned_order_ids: List[int] = field(default_factory=list)

    # Plan / navegación
    waypoints: List[Node] = field(default_factory=list)
    waypoint_idx: int = 0
    route: List[Node] = field(default_factory=list)

    # Estado pickup/drop
    has_picked_up: bool = False
    delivery_queue: List[int] = field(default_factory=list)

    # Fatiga avanzada
    resting: bool = False  # si está descansando (bloquea movimiento)

    # Batching: ticks que esperará en restaurante antes de salir (0 = sin espera)
    wait_until: int = 0

    def can_take_more(self) -> bool:
        """Verifica si el rider tiene capacidad para más pedidos."""
        return len(self.assigned_order_ids) < self.capacity

    def has_task(self) -> bool:
        """Verifica si el rider tiene alguna tarea activa (pedidos o ruta)."""
        return len(self.assigned_order_ids) > 0 or len(self.waypoints) > 0 or len(self.route) > 0

    def current_target(self) -> Optional[Node]:
        """Obtiene el objetivo actual hacia el que se dirige el rider."""
        if not self.waypoints:
            return None
        if self.waypoint_idx < 0 or self.waypoint_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.waypoint_idx]


class FleetManager:
    """Gestor centralizado de la flota de riders.

    Se encarga de la creación, almacenamiento y recuperación de las instancias de Rider.

    Attributes:
        _riders: Lista interna de todos los riders.
        _next_id: Contador para asignar IDs únicos.
    """

    def __init__(self):
        self._riders: List[Rider] = []
        self._next_id = 1

    def add_rider(self, position: Node, speed: float = 1.0) -> Rider:
        """Crea y registra un nuevo rider en la flota.

        Args:
            position: Posición inicial.
            speed: Velocidad base.

        Returns:
            La instancia del Rider creado.
        """
        r = Rider(rider_id=self._next_id, position=position, speed=speed, available=True)
        self._next_id += 1
        self._riders.append(r)
        return r

    def get_all(self) -> List[Rider]:
        """Devuelve una copia de la lista de todos los riders."""
        return list(self._riders)

    def get_available_riders(self) -> List[Rider]:
        """Obtiene riders elegibles para nuevas asignaciones.

        Filtra riders que están descansando. Incluye riders ocupados pero
        con capacidad (batching).

        Returns:
            Lista de Riders disponibles.
        """
        # “available” aquí significa “puede recibir asignación”
        # (si está descansando, NO queremos asignarle más)
        result: List[Rider] = []
        for r in self._riders:
            if r.resting:
                continue
            if r.available:
                result.append(r)
                continue
            if r.wait_until > 0 and r.can_take_more():
                result.append(r)
        return result

