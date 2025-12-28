# core/assignment_engine.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import itertools

from core.route_planner import RoutePlanner
from core.order_manager import Order
from core.fleet_manager import Rider

Node = tuple[int, int]


class AssignmentEngine:
    def __init__(self, planner: RoutePlanner, restaurant_pos: Node):
        self.planner = planner
        self.restaurant_pos = restaurant_pos
        # Hiperparámetros de batching
        self.max_insertion_delta: float = 25.0
        self.slack_tolerance: float = 3.0
        self.activation_cost: float = 5.0

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

    def _path_cost(self, a: Node, b: Node) -> float:
        """Coste real usando A* (determinista)."""
        if a == b:
            return 0.0
        _, c = self.planner.astar(a, b)
        return float(c)

    def _best_route_cost(
        self, rider: Rider, orders: List[Order]
    ) -> Tuple[float, List[Node], List[int]]:
        """
        Devuelve (coste, waypoints, delivery_queue) para la mejor ruta factible
        respetando capacidad (<=3) y precedencia pickup -> dropoff.
        Siempre termina en restaurante para mantener ciclo consistente.
        """
        pending_orders = [o for o in orders if o.is_pending()]
        if not pending_orders:
            return 0.0, [], []

        drop_list = [(o.order_id, o.dropoff) for o in pending_orders]
        best_cost = float("inf")
        best_perm: Optional[Tuple[Tuple[int, Node], ...]] = None

        need_pickup = not rider.has_picked_up

        for perm in itertools.permutations(drop_list):
            cost = 0.0
            current = rider.position
            if need_pickup:
                cost += self._path_cost(current, self.restaurant_pos)
                current = self.restaurant_pos

            for _, drop in perm:
                cost += self._path_cost(current, drop)
                current = drop

            cost += self._path_cost(current, self.restaurant_pos)

            perm_key = tuple(pid for pid, _ in perm)
            if cost < best_cost or (cost == best_cost and perm_key < tuple(
                pid for pid, _ in (best_perm or ())
            )):
                best_cost = cost
                best_perm = perm

        if best_perm is None:
            return float("inf"), [], []

        waypoints: List[Node] = []
        if need_pickup:
            waypoints.append(self.restaurant_pos)
        waypoints.extend([drop for _, drop in best_perm])
        waypoints.append(self.restaurant_pos)

        delivery_queue = [pid for pid, _ in best_perm]
        return best_cost, waypoints, delivery_queue

    def best_plan_for_rider(
        self, rider: Rider, orders: List[Order]
    ) -> Tuple[List[Node], List[int]]:
        """Helper para reconstruir plan con mejor secuencia de drops."""
        _, waypoints, dq = self._best_route_cost(rider, orders)
        return waypoints, dq

    def _delta_cost_for_candidate(
        self, rider: Rider, current_orders: List[Order], candidate: Order
    ) -> float:
        base_cost, _, _ = self._best_route_cost(rider, current_orders)
        with_cost, _, _ = self._best_route_cost(rider, current_orders + [candidate])
        return with_cost - base_cost

    def _pick_best(
        self, orders: List[Order], riders: List[Rider], now: int, urgent_only: bool
    ) -> Optional[Tuple[Order, Rider]]:
        orders_map: Dict[int, Order] = {o.order_id: o for o in orders}
        unassigned = [
            o
            for o in orders
            if o.assigned_to is None
            and (not urgent_only or o.priority > 1 or (o.deadline - now) <= 8)
        ]
        riders = self._eligible_riders(riders)
        if not unassigned or not riders:
            return None

        candidates = []
        for o in unassigned:
            slack = o.deadline - now
            for r in riders:
                assigned_orders = [
                    orders_map[oid]
                    for oid in r.assigned_order_ids
                    if oid in orders_map and orders_map[oid].is_pending()
                ]
                if len(assigned_orders) >= r.capacity:
                    continue

                delta = self._delta_cost_for_candidate(r, assigned_orders, o)
                if delta == float("inf"):
                    continue

                is_partial = len(assigned_orders) > 0
                if is_partial and delta > self.max_insertion_delta:
                    continue
                if is_partial and (slack - delta) < -self.slack_tolerance:
                    continue

                activation_penalty = self.activation_cost if len(assigned_orders) == 0 else 0.0
                effective_cost = delta + activation_penalty

                candidates.append(
                    (
                        slack,
                        effective_cost,
                        r.rider_id,
                        o.order_id,
                        o,
                        r,
                    )
                )

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        _, _, _, _, order, rider = candidates[0]
        return order, rider

    # -----------------------
    # Picks (OPTIMIZADOS con Octile)
    # -----------------------
    def pick_any_nearest(
        self, orders: List[Order], riders: List[Rider], now: int = 0
    ) -> Optional[Tuple[Order, Rider]]:
        """Selecciona el mejor par (order, rider) minimizando coste incremental."""
        return self._pick_best(orders, riders, now=now, urgent_only=False)

    def pick_urgent_nearest(
        self, orders: List[Order], riders: List[Rider], now: int
    ) -> Optional[Tuple[Order, Rider]]:
        """Selecciona el mejor par urgente minimizando coste incremental."""
        return self._pick_best(orders, riders, now=now, urgent_only=True)

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
        # Si está en restaurante, obligamos a nuevo pickup (para batching correcto)
        if rider.position == self.restaurant_pos:
            rider.has_picked_up = False

    def replan_current_leg(self, rider: Rider) -> None:
        tgt = rider.current_target()
        if tgt is None:
            return
        path, _ = self.planner.astar(rider.position, tgt)
        rider.route = path[1:] if path else []
