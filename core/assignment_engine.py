# core/assignment_engine.py
from __future__ import annotations
from typing import List, Optional, Tuple
from core.order_manager import Order
from core.fleet_manager import Rider
from core.route_planner import RoutePlanner

Node = Tuple[int, int]

class AssignmentEngine:
    def __init__(self, planner: RoutePlanner, restaurant_pos: Node):
        self.planner = planner
        self.restaurant_pos = restaurant_pos

    def _eta_to_dropoff_via_restaurant(self, rider: Rider, order: Order) -> float:
        # ETA aproximada: rider->rest + rest->drop
        _, c1 = self.planner.astar(rider.position, order.pickup)
        _, c2 = self.planner.astar(order.pickup, order.dropoff)
        return (c1 + c2) * rider.speed

    def pick_urgent_nearest(self, orders: List[Order], riders: List[Rider], now: int):
        urgent = [o for o in orders if o.is_pending() and (o.is_urgent(now) or o.priority > 1) and o.assigned_to is None]
        if not urgent or not riders:
            return None

        best = None
        best_eta = float("inf")
        for o in urgent:
            for r in riders:
                eta = self._eta_to_dropoff_via_restaurant(r, o)
                if eta < best_eta:
                    best_eta = eta
                    best = (o, r)
        return best  # (order, rider) o None

    def pick_any_nearest(self, orders: List[Order], riders: List[Rider]):
        unassigned = [o for o in orders if o.is_pending() and o.assigned_to is None]
        if not unassigned or not riders:
            return None

        best = None
        best_eta = float("inf")
        for o in unassigned:
            for r in riders:
                eta = self._eta_to_dropoff_via_restaurant(r, o)
                if eta < best_eta:
                    best_eta = eta
                    best = (o, r)
        return best

    def _plan_route_to_target(self, rider: Rider, target: Node) -> List[Node]:
        path, cost = self.planner.astar(rider.position, target)
        if not path:
            return []
        return path[1:]  # quitamos nodo actual

    def assign(self, order: Order, rider: Rider) -> bool:
        """
        Asigna un pedido:
        - rider recibirá plan [rest, cliente, rest]
        - inicialmente va hacia el restaurante (si ya está allí, pasa al cliente)
        """
        order.assigned_to = rider.rider_id
        rider.assigned_order_id = order.order_id
        rider.has_picked_up = False
        rider.available = False

        rider.waypoints = [order.pickup, order.dropoff, order.pickup]
        rider.waypoint_idx = 0

        # Si ya está en el restaurante, "recoge" al instante y va al cliente
        if rider.position == order.pickup:
            rider.has_picked_up = True
            rider.waypoint_idx = 1

        target = rider.current_target()
        if target is None:
            return False

        rider.route = self._plan_route_to_target(rider, target)
        return True

    def replan_current_leg(self, rider: Rider) -> None:
        target = rider.current_target()
        if target is None:
            rider.route = []
            return
        rider.route = self._plan_route_to_target(rider, target)
