# core/assignment_engine.py
from __future__ import annotations
from typing import List, Optional, Tuple

from core.route_planner import RoutePlanner
from core.order_manager import Order
from core.fleet_manager import Rider

Node = tuple[int, int]


class AssignmentEngine:
    def __init__(self, planner: RoutePlanner, restaurant_pos: Node):
        self.planner = planner
        self.restaurant_pos = restaurant_pos

    # -----------------------
    # ETA helpers
    # -----------------------
    def _eta_restaurant_to_drop(self, order: Order) -> float:
        _, c = self.planner.astar(self.restaurant_pos, order.dropoff)
        return float(c)

    def _eta_rider_to_restaurant_plus_drop(self, rider: Rider, order: Order) -> float:
        _, c1 = self.planner.astar(rider.position, self.restaurant_pos)
        _, c2 = self.planner.astar(self.restaurant_pos, order.dropoff)
        return float(c1 + c2)

    def _eligible_riders(self, riders: List[Rider]) -> List[Rider]:
        """
        Fase 2 (simple y estable):
        - Un rider puede coger pedidos SOLO si:
          a) tiene hueco (can_take_more)
          b) y (está disponible) o (está en el restaurante)
        Esto evita que “cargue” pedidos mientras está en ruta.
        """
        out: List[Rider] = []
        for r in riders:
            if not r.can_take_more():
                continue
            if r.available:
                out.append(r)
                continue
            if r.position == self.restaurant_pos:
                out.append(r)
        return out

    # -----------------------
    # Picks
    # -----------------------
    def pick_any_nearest(self, orders: List[Order], riders: List[Rider]) -> Optional[Tuple[Order, Rider]]:
        orders = [o for o in orders if o.assigned_to is None]
        riders = self._eligible_riders(riders)
        if not orders or not riders:
            return None

        best = None
        best_eta = float("inf")

        for o in orders:
            for r in riders:
                if r.position == self.restaurant_pos:
                    eta = self._eta_restaurant_to_drop(o)
                else:
                    eta = self._eta_rider_to_restaurant_plus_drop(r, o)
                if eta < best_eta:
                    best_eta = eta
                    best = (o, r)
        return best

    def pick_urgent_nearest(self, orders: List[Order], riders: List[Rider], now: int) -> Optional[Tuple[Order, Rider]]:
        urgent = [o for o in orders if o.assigned_to is None and (o.priority > 1 or o.is_urgent(now))]
        riders = self._eligible_riders(riders)
        if not urgent or not riders:
            return None

        best = None
        best_eta = float("inf")

        for o in urgent:
            for r in riders:
                if r.position == self.restaurant_pos:
                    eta = self._eta_restaurant_to_drop(o)
                else:
                    eta = self._eta_rider_to_restaurant_plus_drop(r, o)
                if eta < best_eta:
                    best_eta = eta
                    best = (o, r)
        return best

    # -----------------------
    # Assign
    # -----------------------
    def assign(self, order: Order, rider: Rider) -> None:
        if not rider.can_take_more():
            return
        order.assigned_to = rider.rider_id
        if order.order_id not in rider.assigned_order_ids:
            rider.assigned_order_ids.append(order.order_id)
        rider.available = False

    def replan_current_leg(self, rider: Rider) -> None:
        tgt = rider.current_target()
        if tgt is None:
            return
        path, _ = self.planner.astar(rider.position, tgt)
        rider.route = path[1:] if path else []
