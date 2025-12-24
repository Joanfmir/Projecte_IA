# core/fleet_manager.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Node = Tuple[int, int]


@dataclass
class Rider:
    rider_id: int
    position: Node
    speed: float = 1.0  # (por ahora no lo usamos como “pasos por tick”, pero lo dejamos)
    available: bool = True

    # --- Métricas ---
    fatigue: float = 0.0
    distance_travelled: float = 0.0
    deliveries_done: int = 0

    # --- Fase 2: 2 pedidos a la vez ---
    capacity: int = 2
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
        r = Rider(rider_id=self._next_id, position=position, speed=speed, available=True)
        self._next_id += 1
        self._riders.append(r)
        return r

    def get_all(self) -> List[Rider]:
        return list(self._riders)

    def get_available_riders(self) -> List[Rider]:
        # “available” aquí significa “puede recibir asignación”
        # (si está descansando, NO queremos asignarle más)
        return [r for r in self._riders if r.available and (not r.resting)]
