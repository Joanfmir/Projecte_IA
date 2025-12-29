# core/fleet_manager.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Node = Tuple[int, int]

# Nombres para los riders
RIDER_NAMES = [
    "Joan", "Rodolfo", "Pau", "Xènia", "Ignàsi"
]


@dataclass
class Rider:
    rider_id: int
    position: Node
    name: str = ""
    speed: float = 1.0
    available: bool = True

    # --- Métricas ---
    distance_travelled: float = 0.0
    deliveries_done: int = 0

    # --- Fase 2: 3 pedidos a la vez ---
    capacity: int = 3
    assigned_order_ids: List[int] = field(default_factory=list)

    # Plan / navegación
    waypoints: List[Node] = field(default_factory=list)
    waypoint_idx: int = 0
    route: List[Node] = field(default_factory=list)

    # Estado pickup/drop
    has_picked_up: bool = False
    delivery_queue: List[int] = field(default_factory=list)

    # Batching: esperar antes de salir para agrupar pedidos
    wait_until: int = 0  # tick hasta el que debe esperar (0 = no espera)

    def can_take_more(self) -> bool:
        return len(self.assigned_order_ids) < self.capacity

    def has_task(self) -> bool:
        return len(self.assigned_order_ids) > 0 or len(self.waypoints) > 0 or len(self.route) > 0

    def current_target(self) -> Optional[Node]:
        if not self.waypoints:
            return None
        if self.waypoint_idx < 0 or self.waypoint_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.waypoint_idx]


class FleetManager:
    def __init__(self):
        self._riders: List[Rider] = []
        self._next_id = 1

    def add_rider(self, position: Node, speed: float = 1.0) -> Rider:
        name = RIDER_NAMES[(self._next_id - 1) % len(RIDER_NAMES)]
        r = Rider(rider_id=self._next_id, position=position, name=name, speed=speed, available=True)
        self._next_id += 1
        self._riders.append(r)
        return r

    def get_all(self) -> List[Rider]:
        return list(self._riders)

    def get_available_riders(self) -> List[Rider]:
        # Devuelve riders que pueden recibir asignaciones:
        # - Riders disponibles (available=True)
        # - Riders esperando en restaurante para batching (wait_until > 0, pueden tomar más)
        result = []
        for r in self._riders:
            if r.available:
                result.append(r)
            elif r.wait_until > 0 and r.can_take_more():
                # Rider esperando en restaurante, puede recibir más pedidos
                result.append(r)
        return result
