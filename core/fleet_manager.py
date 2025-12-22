# core/fleet_manager.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Node = Tuple[int, int]

@dataclass
class Rider:
    rider_id: int
    position: Node
    speed: float = 1.0
    fatigue: float = 0.0
    available: bool = True

    # Ruta actual (lista de nodos por donde caminar)
    route: List[Node] = field(default_factory=list)

    # Pedido asignado (existe aunque aún no se haya recogido)
    assigned_order_id: Optional[int] = None

    # Indica si ya ha recogido físicamente el pedido en el restaurante
    has_picked_up: bool = False

    # Plan de waypoints: [pickup(rest), dropoff(cliente), pickup(rest)]
    waypoints: List[Node] = field(default_factory=list)
    waypoint_idx: int = 0  # target actual = waypoints[waypoint_idx]

    deliveries_done: int = 0
    distance_travelled: float = 0.0

    def has_task(self) -> bool:
        return self.assigned_order_id is not None

    def current_target(self) -> Optional[Node]:
        if 0 <= self.waypoint_idx < len(self.waypoints):
            return self.waypoints[self.waypoint_idx]
        return None


class FleetManager:
    def __init__(self):
        self.riders: List[Rider] = []
        self._next_id = 1

    def add_rider(self, position: Node, speed: float = 1.0) -> Rider:
        r = Rider(rider_id=self._next_id, position=position, speed=speed)
        self._next_id += 1
        self.riders.append(r)
        return r

    def get_available_riders(self) -> List[Rider]:
        return [r for r in self.riders if r.available and not r.has_task()]

    def get_all(self) -> List[Rider]:
        return self.riders
