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
    # Distancia Octile (O(1) - precisa para grids 8-direccionales)
    # -----------------------
    def _octile(self, a: Node, b: Node) -> float:
        """
        Distancia Octile: heurística óptima para grids con movimiento 8-direccional.
        Diagonal cuesta sqrt(2) ≈ 1.414, ortogonal cuesta 1.
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        # Fórmula: max(dx,dy) + (sqrt(2)-1) * min(dx,dy)
        return max(dx, dy) + 0.414 * min(dx, dy)

    def _eta_octile_restaurant_to_drop(self, order: Order) -> float:
        """ETA aproximada: restaurante -> dropoff."""
        return self._octile(self.restaurant_pos, order.dropoff)

    def _eta_octile_rider_to_drop(self, rider: Rider, order: Order) -> float:
        """ETA aproximada: rider -> restaurante -> dropoff."""
        return self._octile(rider.position, self.restaurant_pos) + self._octile(
            self.restaurant_pos, order.dropoff
        )

    # -----------------------
    # ETA helpers con A* (solo para precisión final si se necesita)
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
        Criterios de elegibilidad (F3 FIX):
        - can_take_more() = tiene capacidad
        - NOT resting = no está descansando
        - (available OR en restaurante) = listo para recibir pedido
        """
        out: List[Rider] = []
        for r in riders:
            if not r.can_take_more():
                continue
            # F3 FIX: Excluir riders descansando
            if getattr(r, "resting", False):
                continue
            if r.available:
                out.append(r)
                continue
            if r.position == self.restaurant_pos:
                out.append(r)
        return out

    # -----------------------
    # Picks (OPTIMIZADOS con Octile)
    # -----------------------
    def pick_any_nearest(
        self, orders: List[Order], riders: List[Rider]
    ) -> Optional[Tuple[Order, Rider]]:
        """Selecciona el mejor par (order, rider) usando Octile heuristic."""
        orders = [o for o in orders if o.assigned_to is None]
        riders = self._eligible_riders(riders)
        if not orders or not riders:
            return None

        best = None
        best_eta = float("inf")

        for o in orders:
            for r in riders:
                # Usar Octile: O(1) en vez de A*: O(N log N)
                if r.position == self.restaurant_pos:
                    eta = self._eta_octile_restaurant_to_drop(o)
                else:
                    eta = self._eta_octile_rider_to_drop(r, o)
                if eta < best_eta:
                    best_eta = eta
                    best = (o, r)
        return best

    def pick_urgent_nearest(
        self, orders: List[Order], riders: List[Rider], now: int
    ) -> Optional[Tuple[Order, Rider]]:
        """Selecciona el mejor par urgente (order, rider) usando Octile heuristic."""
        # URG_SLACK alineado con factored_states.extract_features()
        URG_SLACK = 8
        urgent = [
            o
            for o in orders
            if o.assigned_to is None
            and (o.priority > 1 or (o.deadline - now) <= URG_SLACK)
        ]
        riders = self._eligible_riders(riders)
        if not urgent or not riders:
            return None

        best = None
        best_eta = float("inf")

        for o in urgent:
            for r in riders:
                # Usar Octile: O(1) en vez de A*: O(N log N)
                if r.position == self.restaurant_pos:
                    eta = self._eta_octile_restaurant_to_drop(o)
                else:
                    eta = self._eta_octile_rider_to_drop(r, o)
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
