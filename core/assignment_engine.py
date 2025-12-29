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
        # Hiperparámetros de batching inteligente
        self.max_insertion_delta: float = 25.0  # máx incremento de coste aceptable
        self.slack_tolerance: float = 3.0       # margen de tiempo permitido
        self.activation_cost: float = 2.0       # penalización por activar rider vacío

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

    def _path_cost(self, a: Node, b: Node) -> float:
        """Coste real usando A* (determinista)."""
        if a == b:
            return 0.0
        _, c = self.planner.astar(a, b)
        return float(c)

    def _eligible_riders(self, riders: List[Rider]) -> List[Rider]:
        """
        Un rider puede coger pedidos SOLO si:
          a) tiene hueco (can_take_more)
          b) está FÍSICAMENTE en el restaurante (necesita la pizza!)
        """
        out: List[Rider] = []
        for r in riders:
            if not r.can_take_more():
                continue
            # SOLO puede recibir pedidos si está en el restaurante
            if r.position == self.restaurant_pos:
                out.append(r)
        return out

    # -----------------------
    # Cálculo de mejor ruta (TSP pequeño)
    # -----------------------
    def _best_route_cost(
        self, rider: Rider, orders: List[Order]
    ) -> Tuple[float, List[Node], List[int]]:
        """
        Devuelve (coste, waypoints, delivery_queue) para la mejor ruta factible.
        Prueba todas las permutaciones (OK porque capacity <= 3).
        """
        pending_orders = [o for o in orders if o.is_pending()]
        if not pending_orders:
            return 0.0, [], []

        drop_list = [(o.order_id, o.dropoff) for o in pending_orders]
        best_cost = float("inf")
        best_perm: Optional[Tuple[Tuple[int, Node], ...]] = None
        EPS = 1e-6

        need_pickup = not rider.has_picked_up

        # Brute-force permutations. O(n!) pero n <= 3, así que máximo 6 permutaciones
        for perm in itertools.permutations(drop_list):
            cost = 0.0
            current = rider.position
            
            if need_pickup:
                cost += self._path_cost(current, self.restaurant_pos)
                current = self.restaurant_pos

            for _, drop in perm:
                cost += self._path_cost(current, drop)
                current = drop

            # Volver al restaurante
            cost += self._path_cost(current, self.restaurant_pos)

            if cost < best_cost - EPS:
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

    def _delta_cost_for_candidate(
        self, rider: Rider, current_orders: List[Order], candidate: Order
    ) -> float:
        """
        Calcula cuánto AUMENTA el coste de la ruta al añadir un nuevo pedido.
        Esto es más inteligente que la distancia simple porque considera
        la posición relativa de todos los drops.
        """
        base_cost, _, _ = self._best_route_cost(rider, current_orders)
        with_cost, _, _ = self._best_route_cost(rider, current_orders + [candidate])
        return with_cost - base_cost

    def _distance_to_existing_orders(self, rider: Rider, new_order: Order, all_orders: List[Order]) -> float:
        """
        Calcula la distancia mínima del nuevo pedido a los pedidos que ya tiene asignados el rider.
        Usado para batching: preferir riders con pedidos cercanos.
        """
        if not rider.assigned_order_ids:
            return float("inf")
        
        min_dist = float("inf")
        for oid in rider.assigned_order_ids:
            for o in all_orders:
                if o.order_id == oid:
                    # Distancia Manhattan entre dropoffs
                    dx = abs(new_order.dropoff[0] - o.dropoff[0])
                    dy = abs(new_order.dropoff[1] - o.dropoff[1])
                    dist = dx + dy
                    min_dist = min(min_dist, dist)
                    break
        return min_dist

    # -----------------------
    # Pick con Delta-Cost (batching inteligente)
    # -----------------------
    def _pick_best_delta(
        self, orders: List[Order], riders: List[Rider], now: int, urgent_only: bool
    ) -> Optional[Tuple[Order, Rider]]:
        """
        Selección inteligente usando delta-cost.
        Considera el incremento real de coste al añadir cada pedido.
        """
        orders_map: Dict[int, Order] = {o.order_id: o for o in orders}
        
        # Filtrar pedidos no asignados (y urgentes si aplica)
        unassigned = [
            o for o in orders
            if o.assigned_to is None
            and (not urgent_only or o.priority > 1 or (o.deadline - now) <= 8)
        ]
        
        eligible = self._eligible_riders(riders)
        if not unassigned or not eligible:
            return None

        candidates = []
        for o in unassigned:
            slack = o.deadline - now  # tiempo restante hasta deadline
            
            for r in eligible:
                # Obtener pedidos actuales del rider
                assigned_orders = [
                    orders_map[oid]
                    for oid in r.assigned_order_ids
                    if oid in orders_map and orders_map[oid].is_pending()
                ]
                
                if len(assigned_orders) >= r.capacity:
                    continue

                # Penalización por activar rider vacío (preferir agrupar)
                activation_penalty = self.activation_cost if len(assigned_orders) == 0 else 0.0

                # Calcular delta-cost
                delta = self._delta_cost_for_candidate(r, assigned_orders, o)
                if delta == float("inf"):
                    continue

                # Filtros de viabilidad para batching
                is_partial = len(assigned_orders) > 0
                if is_partial and delta > self.max_insertion_delta:
                    continue  # Incremento demasiado grande
                if is_partial and (slack - delta) < -self.slack_tolerance:
                    continue  # No llegaría a tiempo

                effective_cost = delta + activation_penalty

                candidates.append((
                    slack,           # tiempo restante
                    effective_cost,  # coste efectivo
                    r.rider_id,      # desempate
                    o.order_id,      # desempate
                    o,
                    r,
                ))

        if not candidates:
            return None

        # Ordenar: menor coste efectivo, luego menor slack (más urgente primero)
        candidates.sort(key=lambda x: (x[1], x[0], x[2], x[3]))
        _, _, _, _, order, rider = candidates[0]
        return order, rider

    # -----------------------
    # Picks (usan delta-cost)
    # -----------------------
    def pick_any_nearest(
        self, orders: List[Order], riders: List[Rider], all_orders: List[Order] = None, now: int = 0
    ) -> Optional[Tuple[Order, Rider]]:
        """Selecciona el mejor par (order, rider) minimizando coste incremental."""
        return self._pick_best_delta(orders, riders, now=now, urgent_only=False)

    def pick_urgent_nearest(
        self, orders: List[Order], riders: List[Rider], now: int, all_orders: List[Order] = None
    ) -> Optional[Tuple[Order, Rider]]:
        """Selecciona el mejor par urgente minimizando coste incremental."""
        return self._pick_best_delta(orders, riders, now=now, urgent_only=True)

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
