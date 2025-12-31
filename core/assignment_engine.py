# core/assignment_engine.py
"""Motor de asignación de pedidos a repartidores.

Este módulo contiene la lógica principal para asignar pedidos pendientes a los
riders disponibles, calculando costes de ruta y tiempos de entrega estimados (ETA)
para optimizar la eficiencia de la flota.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import itertools

from core.route_planner import RoutePlanner
from core.order_manager import Order
from core.fleet_manager import Rider

Node = tuple[int, int]


class AssignmentEngine:
    """Motor de asignación que gestiona el emparejamiento de pedidos y riders.

    Utiliza heurísticas de distancia (Octile) y algoritmos de búsqueda (A*)
    para determinar la mejor asignación posible minimizando costes y tiempos.

    Attributes:
        planner: Instancia de RoutePlanner para cálculos de rutas.
        restaurant_pos: Coordenadas (x, y) del restaurante.
        activation_cost: Coste penalización por activar un rider inactivo.
        max_insertion_delta: Máximo coste adicional permitido para insertar un pedido.
        slack_tolerance: Tolerancia para el cumplimiento de deadlines.
    """

    def __init__(
        self, planner: RoutePlanner, restaurant_pos: Node, activation_cost: float = 2.0
    ):
        self.planner = planner
        self.restaurant_pos = restaurant_pos
        # Hiperparámetros de batching
        self.max_insertion_delta: float = 25.0
        self.slack_tolerance: float = 3.0
        self.activation_cost: float = activation_cost

    # -----------------------
    # Distancia Octile (O(1) - precisa para grids 8-direccionales)
    # -----------------------
    def _octile(self, a: Node, b: Node) -> float:
        """Calcula la distancia Octile entre dos puntos.

        Heurística óptima para grids con movimiento en 8 direcciones.
        El movimiento diagonal cuesta sqrt(2) ≈ 1.414, y el ortogonal cuesta 1.

        Args:
            a: Nodo origen (x, y).
            b: Nodo destino (x, y).

        Returns:
            La distancia estimada entre a y b.
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        # Fórmula: max(dx,dy) + (sqrt(2)-1) * min(dx,dy)
        return max(dx, dy) + 0.414 * min(dx, dy)

    def _eta_octile_restaurant_to_drop(self, order: Order) -> float:
        """Calcula ETA aproximada desde el restaurante hasta el punto de entrega.

        Args:
            order: El pedido para el cual calcular la ETA.

        Returns:
            Tiempo estimado usando distancia Octile.
        """
        return self._octile(self.restaurant_pos, order.dropoff)

    def _eta_octile_rider_to_drop(self, rider: Rider, order: Order) -> float:
        """Calcula ETA aproximada: rider -> restaurante -> entrega.

        Args:
            rider: El repartidor.
            order: El pedido objetivo.

        Returns:
            Tiempo total estimado (rider a restaurante + restaurante a entrega).
        """
        return self._octile(rider.position, self.restaurant_pos) + self._octile(
            self.restaurant_pos, order.dropoff
        )

    # -----------------------
    # ETA helpers con A* (solo para precisión final si se necesita)
    # -----------------------
    def _eta_restaurant_to_drop(self, order: Order) -> float:
        """Calcula ETA precisa usando A* desde restaurante a entrega."""
        _, c = self.planner.astar(self.restaurant_pos, order.dropoff)
        return float(c)

    def _eta_rider_to_restaurant_plus_drop(self, rider: Rider, order: Order) -> float:
        """Calcula ETA precisa usando A*: rider -> restaurante -> entrega."""
        _, c1 = self.planner.astar(rider.position, self.restaurant_pos)
        _, c2 = self.planner.astar(self.restaurant_pos, order.dropoff)
        return float(c1 + c2)

    def _eligible_riders(self, riders: List[Rider]) -> List[Rider]:
        """Filtra y devuelve la lista de riders elegibles para asignación.

        Criterios de elegibilidad:
        1. Tiene capacidad disponible (can_take_more).
        2. No está descansando (not resting).
        3. Está disponible O está en el restaurante esperando.

        Args:
            riders: Lista completa de riders.

        Returns:
            Lista de riders que pueden aceptar un nuevo pedido.
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
        """Calcula el coste real del camino entre dos nodos usando A*.

        Args:
            a: Nodo origen.
            b: Nodo destino.

        Returns:
            Coste del camino (float). Retorna 0.0 si a == b.
        """
        if a == b:
            return 0.0
        _, c = self.planner.astar(a, b)
        return float(c)

    def _best_route_cost(
        self, rider: Rider, orders: List[Order]
    ) -> Tuple[float, List[Node], List[int]]:
        """Calcula la mejor ruta para un conjunto de pedidos asignados a un rider.

        Prueba permutaciones de entrega para minimizar el coste total, respetando
        la restricción de recoger en restaurante antes de entregar.
        La ruta siempre termina regresando al restaurante.

        Args:
            rider: El repartidor.
            orders: Lista de pedidos a entregar.

        Returns:
            Tupla con (coste_total, lista_de_waypoints, cola_de_entregas).
            Retorna (inf, [], []) si no hay ruta válida.
        """
        pending_orders = [o for o in orders if o.is_pending()]
        if not pending_orders:
            return 0.0, [], []

        drop_list = [(o.order_id, o.dropoff) for o in pending_orders]
        best_cost = float("inf")
        best_perm: Optional[Tuple[Tuple[int, Node], ...]] = None
        best_key: Optional[Tuple[int, ...]] = None
        EPS = 1e-6

        need_pickup = not rider.has_picked_up

        # NOTA: iteramos permutaciones por fuerza bruta. Complejidad O(n!),
        # pero la capacidad es pequeña (<=3), así que es manejable.
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
            # Mantenemos la mejor permutación encontrada
            if (cost + EPS) < best_cost or (
                abs(cost - best_cost) <= EPS and (best_key is None or perm_key < best_key)
            ):
                best_cost = cost
                best_perm = perm
                best_key = perm_key

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
        """Genera el mejor plan de ruta para un rider dado.

        Args:
            rider: El repartidor.
            orders: Lista de pedidos.

        Returns:
            Tupla (waypoints, delivery_queue) con la mejor secuencia.
        """
        _, waypoints, dq = self._best_route_cost(rider, orders)
        return waypoints, dq

    def _delta_cost_for_candidate(
        self, rider: Rider, current_orders: List[Order], candidate: Order
    ) -> float:
        """Calcula el coste marginal de añadir un pedido candidato a un rider.

        Args:
            rider: El repartidor.
            current_orders: Pedidos actuales del rider.
            candidate: Nuevo pedido potencial.

        Returns:
            Diferencia de coste (nuevo_coste - coste_base).
        """
        base_cost, _, _ = self._best_route_cost(rider, current_orders)
        with_cost, _, _ = self._best_route_cost(rider, current_orders + [candidate])
        return with_cost - base_cost

    def _pick_best(
        self, orders: List[Order], riders: List[Rider], now: int, urgent_only: bool
    ) -> Optional[Tuple[Order, Rider]]:
        """Selecciona el mejor par (pedido, rider) para asignar.

        Evalúa todas las combinaciones posibles de pedidos sin asignar y riders
        elegibles, calculando el coste de inserción y priorizando según slack y coste.

        Args:
            orders: Lista total de pedidos.
            riders: Lista total de riders.
            now: Tiempo actual de la simulación.
            urgent_only: Si es True, solo considera pedidos urgentes o próximos a deadline.

        Returns:
            Tupla (Order, Rider) con la mejor asignación, o None si no hay ninguna válida.
        """
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

                activation_penalty = self.activation_cost if len(assigned_orders) == 0 else 0.0

                delta = self._delta_cost_for_candidate(r, assigned_orders, o)
                if delta == float("inf"):
                    continue

                is_partial = len(assigned_orders) > 0
                if is_partial and delta > self.max_insertion_delta:
                    continue
                if is_partial and (slack - delta) < -self.slack_tolerance:
                    continue

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

        # Slack ya fue filtrado; priorizar coste efectivo, luego slack, rider, pedido
        candidates.sort(key=lambda x: (x[1], x[0], x[2], x[3]))
        _, _, _, _, order, rider = candidates[0]
        return order, rider

    # -----------------------
    # Métodos de Selección (Picks)
    # -----------------------
    def pick_any_nearest(
        self, orders: List[Order], riders: List[Rider], now: int = 0
    ) -> Optional[Tuple[Order, Rider]]:
        """Selecciona el mejor par (pedido, rider) minimizando coste incremental.

        Args:
            orders: Lista de pedidos.
            riders: Lista de riders.
            now: Tiempo actual.

        Returns:
            La mejor asignación disponible o None.
        """
        return self._pick_best(orders, riders, now=now, urgent_only=False)

    def pick_urgent_nearest(
        self, orders: List[Order], riders: List[Rider], now: int
    ) -> Optional[Tuple[Order, Rider]]:
        """Selecciona el mejor par urgente minimizando coste incremental.

        Args:
            orders: Lista de pedidos.
            riders: Lista de riders.
            now: Tiempo actual.

        Returns:
            La mejor asignación urgente o None.
        """
        return self._pick_best(orders, riders, now=now, urgent_only=True)

    # -----------------------
    # Asignación y Replanificación
    # -----------------------
    def assign(self, order: Order, rider: Rider) -> None:
        """Asigna formalmente un pedido a un rider y actualiza su estado.

        Args:
            order: El pedido a asignar.
            rider: El repartidor seleccionado.
        """
        if not rider.can_take_more():
            return
        order.assigned_to = rider.rider_id
        if order.order_id not in rider.assigned_order_ids:
            rider.assigned_order_ids.append(order.order_id)
        rider.available = False
        # Si está en restaurante, obligamos a nuevo pickup (para batching correcto)
        if rider.position == self.restaurant_pos and not getattr(rider, "delivery_queue", []):
            rider.has_picked_up = False

    def replan_current_leg(self, rider: Rider) -> None:
        """Recalcula la ruta del tramo actual del rider.

        Útil cuando ocurren cambios en el tráfico o bloqueos. Actualiza
        rider.route con el nuevo camino óptimo hacia el objetivo actual.

        Args:
            rider: El repartidor a replanificar.
        """
        tgt = rider.current_target()
        if tgt is None:
            return
        path, _ = self.planner.astar(rider.position, tgt)
        rider.route = path[1:] if path else []

