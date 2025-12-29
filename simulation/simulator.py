# simulation/simulator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List, Set
import random
import statistics
import math

from core.road_graph import RoadGraph
from core.route_planner import RoutePlanner
from core.order_manager import OrderManager, Order
from core.fleet_manager import FleetManager, Rider
from core.assignment_engine import AssignmentEngine
from core.dispatch_policy import (
    make_state,
    A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC
)

Node = Tuple[int, int]


@dataclass
class SimConfig:
    width: int = 25
    height: int = 25
    n_riders: int = 4
    episode_len: int = 300
    order_spawn_prob: float = 0.15
    max_eta: int = 70
    seed: int = 7

    # urban layout
    block_size: int = 5
    street_width: int = 1

    # cierres de calles
    road_closure_prob: float = 0.05
    road_closures_per_event: int = 1

    # batching: tiempo de espera en restaurante para agrupar pedidos
    batch_wait_ticks: int = 5  # ticks que espera antes de salir


class Simulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        self.avenues: List[dict] = []
        self.buildings: Set[Node] = self._generate_urban_buildings()

        self.graph = RoadGraph(cfg.width, cfg.height, base_cost=1.0, seed=cfg.seed, blocked=self.buildings)
        self.planner = RoutePlanner(self.graph)

        self.om = OrderManager()
        self.fm = FleetManager()

        self.restaurant: Node = self._nearest_walkable((cfg.width // 2, cfg.height // 2))
        self.assigner = AssignmentEngine(self.planner, restaurant_pos=self.restaurant)

        self.t = 0
        self.traffic_level = "low"

        for _ in range(cfg.n_riders):
            sp = self.rng.choice([0.9, 1.0, 1.1])
            r = self.fm.add_rider(position=self.restaurant, speed=sp)
            r.available = True

    def reset(self) -> None:
        self.__init__(self.cfg)

    # -----------------------
    # URBAN GENERATION
    # -----------------------
    def _generate_urban_buildings(self) -> Set[Node]:
        W, H = self.cfg.width, self.cfg.height
        bs = self.cfg.block_size
        sw = self.cfg.street_width

        buildings: Set[Node] = set()
        self.avenues = []

        step = bs + sw
        for bx in range(0, W, step):
            for by in range(0, H, step):
                for x in range(bx, min(bx + bs, W)):
                    for y in range(by, min(by + bs, H)):
                        buildings.add((x, y))

        for x in range(W):
            buildings.discard((x, 0))
            buildings.discard((x, H - 1))
        for y in range(H):
            buildings.discard((0, y))
            buildings.discard((W - 1, y))

        n_avenues = self.rng.choice([1, 2, 2, 3])
        avenue_width = max(1, sw)

        possible_slopes = [0.5, 0.75, 1.0, 1.25, 1.5, -0.5, -0.75, -1.0, -1.25, -1.5]
        used = set()

        for _ in range(n_avenues):
            m = self.rng.choice(possible_slopes)
            while m in used and len(used) < len(possible_slopes):
                m = self.rng.choice(possible_slopes)
            used.add(m)

            b = self.rng.uniform(-H * 0.5, H * 1.5)
            self.avenues.append({"m": m, "b": b, "w": avenue_width})

            for x in range(W):
                y_round = int(round(m * x + b))
                for dy in range(-avenue_width, avenue_width + 1):
                    y = y_round + dy
                    if 0 <= y < H:
                        buildings.discard((x, y))

            if abs(m) > 0.01:
                for y in range(H):
                    x_round = int(round((y - b) / m))
                    for dx in range(-avenue_width, avenue_width + 1):
                        x = x_round + dx
                        if 0 <= x < W:
                            buildings.discard((x, y))

        return buildings

    def _nearest_walkable(self, start: Node) -> Node:
        if self.graph.is_walkable(start):
            return start

        from collections import deque
        q = deque([start])
        seen = {start}

        while q:
            x, y = q.popleft()
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nb = (x+dx, y+dy)
                if nb in seen:
                    continue
                if 0 <= nb[0] < self.cfg.width and 0 <= nb[1] < self.cfg.height:
                    if self.graph.is_walkable(nb):
                        return nb
                    seen.add(nb)
                    q.append(nb)

        return (self.cfg.width // 2, self.cfg.height // 2)

    def _random_walkable_cell(self) -> Optional[Node]:
        for _ in range(6000):
            x = self.rng.randrange(self.cfg.width)
            y = self.rng.randrange(self.cfg.height)
            cell = (x, y)
            if cell == self.restaurant:
                continue
            if not self.graph.is_walkable(cell):
                continue
            path, cost = self.planner.astar(self.restaurant, cell)
            if path and cost != float("inf") and len(path) >= 2:
                return cell
        return None

    # -------------------
    # PLANIFICACIÓN
    # -------------------
    def _get_order(self, order_id: int) -> Optional[Order]:
        return self.om.get_order(order_id)

    def _sorted_drop_queue(self, rider: Rider) -> List[int]:
        valid: List[Order] = []
        for oid in rider.assigned_order_ids:
            o = self._get_order(oid)
            if o is not None and o.is_pending():
                if o.dropoff == self.restaurant:
                    continue
                valid.append(o)

        if not valid:
            return []

        scored = []
        for o in valid:
            _, c = self.planner.astar(self.restaurant, o.dropoff)
            scored.append((float(c), o.order_id))
        scored.sort(key=lambda x: x[0])
        return [oid for _, oid in scored]

    def _try_batch_nearby_orders(self, rider: Rider) -> None:
        """
        Intenta asignar pedidos cercanos al rider antes de que salga.
        Usado cuando el rider va a salir inmediatamente (pedido ya esperaba).
        """
        BATCH_DISTANCE_THRESHOLD = 14  # Misma distancia que en assignment_engine
        
        # Obtener pedidos pendientes sin asignar
        pending_unassigned = [o for o in self.om.get_pending_orders() if o.assigned_to is None]
        
        if not pending_unassigned:
            return
        
        # Obtener dropoffs de los pedidos que ya tiene el rider
        existing_dropoffs = []
        for oid in rider.assigned_order_ids:
            o = self._get_order(oid)
            if o is not None:
                existing_dropoffs.append(o.dropoff)
        
        if not existing_dropoffs:
            return
        
        # Buscar pedidos cercanos a los que ya tiene
        for o in pending_unassigned:
            if not rider.can_take_more():
                break
            
            # Calcular distancia mínima a pedidos existentes
            min_dist = float("inf")
            for drop in existing_dropoffs:
                dist = abs(o.dropoff[0] - drop[0]) + abs(o.dropoff[1] - drop[1])
                min_dist = min(min_dist, dist)
            
            # Si está cerca, asignarlo
            if min_dist <= BATCH_DISTANCE_THRESHOLD:
                self.assigner.assign(o, rider)
                existing_dropoffs.append(o.dropoff)

    def _rebuild_plan_for_rider(self, rider: Rider) -> None:
        rider.assigned_order_ids = [
            oid for oid in rider.assigned_order_ids
            if (self._get_order(oid) and self._get_order(oid).is_pending())
        ]

        if not rider.assigned_order_ids:
            rider.delivery_queue = []
            rider.has_picked_up = False

            if rider.position != self.restaurant:
                rider.available = False
                rider.waypoints = [self.restaurant]
                rider.waypoint_idx = 0
                path, _ = self.planner.astar(rider.position, self.restaurant)
                rider.route = path[1:] if path else []
            else:
                rider.available = True
                rider.waypoints = []
                rider.waypoint_idx = 0
                rider.route = []
            return

        rider.available = False
        rider.delivery_queue = self._sorted_drop_queue(rider)

        waypoints: List[Node] = []

        if not rider.has_picked_up:
            waypoints.append(self.restaurant)
            # Batching: si está en el restaurante, iniciar o mantener espera
            if rider.position == self.restaurant:
                # Comprobar si algún pedido ya estaba esperando (creado hace más de 2 ticks)
                any_order_was_waiting = False
                for oid in rider.assigned_order_ids:
                    o = self._get_order(oid)
                    if o is not None and (self.t - o.created_at) > 2:
                        any_order_was_waiting = True
                        break
                
                if any_order_was_waiting:
                    # El pedido ya esperó, pero antes de salir, intentar coger más pedidos cercanos
                    if rider.can_take_more():
                        self._try_batch_nearby_orders(rider)
                    # Ahora sí, salir inmediatamente
                    rider.wait_until = 0
                elif rider.can_take_more():
                    # Pedido recién creado y tiene espacio, iniciar/mantener espera
                    if rider.wait_until == 0:
                        rider.wait_until = self.t + self.cfg.batch_wait_ticks
                else:
                    # Ya está lleno (2 pedidos), salir inmediatamente
                    rider.wait_until = 0

        # Recalcular delivery_queue por si se añadieron pedidos
        rider.delivery_queue = self._sorted_drop_queue(rider)

        for oid in rider.delivery_queue:
            o = self._get_order(oid)
            if o is not None:
                waypoints.append(o.dropoff)

        waypoints.append(self.restaurant)

        rider.waypoints = waypoints
        rider.waypoint_idx = 0

        tgt = rider.current_target()
        if tgt is None:
            rider.route = []
            return

        path, _ = self.planner.astar(rider.position, tgt)
        rider.route = path[1:] if path else []

    # -------------------
    # SNAPSHOT
    # -------------------
    def snapshot(self) -> dict:
        pending = self.om.get_pending_orders()
        riders = self.fm.get_all()

        delivered_total = 0
        delivered_ontime = 0
        delivered_late = 0
        for o in self.om.orders:
            if o.delivered_at is not None:
                delivered_total += 1
                if o.delivered_at <= o.deadline:
                    delivered_ontime += 1
                else:
                    delivered_late += 1

        orders_full = []
        for o in self.om.orders:
            if o.delivered_at is not None:
                st = "delivered"
            elif o.picked_up_at is not None:
                st = "picked"
            else:
                st = "pending"

            orders_full.append({
                "id": o.order_id,
                "priority": o.priority,
                "status": st,
                "pickup": o.pickup,
                "dropoff": o.dropoff,
                "created_at": o.created_at,
                "deadline": o.deadline,
                "assigned_to": o.assigned_to,
                "picked_up_at": o.picked_up_at,
                "delivered_at": o.delivered_at,
            })

        return {
            "t": self.t,
            "restaurant": self.restaurant,
            "buildings": list(self.buildings),
            "avenues": list(self.avenues),

            "pending_orders": [(o.dropoff, o.priority, o.deadline, o.assigned_to) for o in pending],
            "orders_full": orders_full,

            "riders": [
                {
                    "id": r.rider_id,
                    "name": r.name,
                    "pos": r.position,
                    "route": list(r.route),

                    "carrying": (r.delivery_queue[0] if (r.has_picked_up and r.delivery_queue) else None),
                    "assigned": list(r.assigned_order_ids),
                    "picked": r.has_picked_up,
                    "waypoints": list(r.waypoints),
                    "wp_idx": r.waypoint_idx,
                    "available": r.available,
                    "resting": bool(getattr(r, "resting", False)),  # ✅ NUEVO
                }
                for r in riders
            ],
            "traffic": self.traffic_level,
            "closures": self.graph.count_closed_directed(),
            "closed_edges": self.graph.get_closed_edges_sample(200),
            "blocked": self.graph.count_blocked(),
            "blocked_nodes": list(self.graph.blocked - self.buildings),  # Solo los cierres dinámicos
            "width": self.cfg.width,
            "height": self.cfg.height,
            "delivered_total": delivered_total,
            "delivered_ontime": delivered_ontime,
            "delivered_late": delivered_late,
        }

    def print_late_orders(self) -> None:
        """Imprime al final los pedidos que llegaron tarde."""
        late_orders = []
        for o in self.om.orders:
            if o.delivered_at is not None and o.delivered_at > o.deadline:
                late_orders.append(o)
        
        if not late_orders:
            print("\n✅ ¡Todos los pedidos llegaron a tiempo!")
            return
        
        print(f"\n❌ PEDIDOS QUE LLEGARON TARDE: {len(late_orders)}")
        print("-" * 60)
        
        # Función para convertir ticks a hora legible
        def tick_to_time(tick: int) -> str:
            total_hours = 5.0  # 19:00 a 00:00
            hours_elapsed = (tick / self.cfg.episode_len) * total_hours
            hour = 19 + int(hours_elapsed)
            minutes = int((hours_elapsed % 1) * 60)
            return f"{hour:02d}:{minutes:02d}"
        
        for o in late_orders:
            created_time = tick_to_time(o.created_at)
            deadline_time = tick_to_time(o.deadline)
            delivered_time = tick_to_time(o.delivered_at)
            retraso = o.delivered_at - o.deadline
            retraso_min = (retraso / self.cfg.episode_len) * 5 * 60  # en minutos
            
            urgente = "URGENTE" if o.priority > 1 else "normal"
            print(f"  Pedido #{o.order_id:3d} ({urgente})")
            print(f"    Creado: {created_time} | Deadline: {deadline_time} | Entregado: {delivered_time}")
            print(f"    Retraso: {retraso} ticks (~{retraso_min:.1f} min)")
            print()

    # -------------------
    # Generación de pedidos / tráfico
    # -------------------
    def _get_spawn_probability(self) -> float:
        """
        Calcula la probabilidad de spawn usando una distribución gaussiana.
        Pico: ~20:30 (1.5 horas desde las 19:00)
        La probabilidad base se modula por la curva gaussiana.
        """
        # Convertir tick actual a "horas desde las 19:00"
        total_hours = 5.0  # 19:00 a 00:00
        hours_elapsed = (self.t / self.cfg.episode_len) * total_hours
        
        # Parámetros de la gaussiana
        peak_hour = 1.5      # Pico a las 20:30 (1.5h desde las 19:00)
        sigma = 1.2          # Desviación estándar (~1.2 horas de ancho)
        
        # Curva gaussiana: máximo 1.0 en el pico, decrece hacia los lados
        gaussian = math.exp(-0.5 * ((hours_elapsed - peak_hour) / sigma) ** 2)
        
        # Escalar: en el pico la prob es ~2x la base, al final ~0.3x
        # Esto mantiene el promedio cercano a la prob base
        min_factor = 0.2
        max_factor = 2.0
        factor = min_factor + (max_factor - min_factor) * gaussian
        
        return self.cfg.order_spawn_prob * factor

    def maybe_spawn_order(self) -> None:
        # No generar más pedidos si queda poco tiempo para que termine
        # Dejar margen para entregar los pendientes (max_eta ticks)
        ticks_remaining = self.cfg.episode_len - self.t
        if ticks_remaining <= self.cfg.max_eta:
            return  # No más pedidos, dejar que entreguen los pendientes

        spawn_prob = self._get_spawn_probability()
        if self.rng.random() < spawn_prob:
            drop = self._random_walkable_cell()
            if drop is None or drop in self.graph.blocked:
                return  # No crear pedidos en zonas bloqueadas
            prio = 2 if self.rng.random() < 0.15 else 1
            max_eta = self.cfg.max_eta if prio == 1 else int(self.cfg.max_eta * 0.6)

            self.om.create_order(
                pickup=self.restaurant,
                dropoff=drop,
                now=self.t,
                max_eta=max_eta,
                priority=prio
            )
    def cancel_orders_in_blocked_zones(self) -> None:
        # Cancelar pedidos cuyo dropoff esté bloqueado
        for o in self.om.get_pending_orders():
            if o.dropoff in self.graph.blocked:
                self.om.cancel_order(o.order_id, reason="bloqueo_calle")

    def maybe_change_traffic(self) -> None:
        if self.t % 60 == 0 and self.t > 0:
            self.traffic_level = self.rng.choice(["low", "medium", "high"])
            self.graph.set_traffic_level(self.traffic_level)

    def maybe_road_closure(self) -> None:
        """Genera cierres de calles aleatorios según road_closure_prob, pero nunca más de 4 activos."""
        # Contar bloqueos activos (solo dinámicos, no edificios)
        active_closures = len(self.graph.blocked - self.buildings)
        if active_closures >= 4:
            return  # No generar más cierres
        if self.rng.random() < self.cfg.road_closure_prob:
            self.graph.random_road_incidents(self.cfg.road_closures_per_event)
            # Replanificar rutas de riders afectados
            for r in self.fm.get_all():
                if r.route:
                    # Verificar si la ruta actual pasa por una calle cerrada
                    current = r.position
                    needs_replan = False
                    for next_pos in r.route:
                        edge = (current, next_pos)
                        if edge in self.graph.edges and self.graph.edges[edge].closed:
                            needs_replan = True
                            break
                        current = next_pos
                    
                    if needs_replan:
                        tgt = r.current_target()
                        if tgt:
                            path, _ = self.planner.astar(r.position, tgt)
                            r.route = path[1:] if path else []

    # -------------------
    # Movimiento
    # -------------------
    def move_riders_one_tick(self) -> List[Order]:
        delivered_now: List[Order] = []

        for r in self.fm.get_all():
            # Si el rider tiene pedidos asignados y alguno está bloqueado, cancelar y volver a la pizzería
            blocked_orders = [oid for oid in r.assigned_order_ids if self._get_order(oid) and self._get_order(oid).dropoff in self.graph.blocked]
            if blocked_orders:
                # Cancelar los pedidos bloqueados
                for oid in blocked_orders:
                    self.om.cancel_order(oid, reason="bloqueo_calle_en_ruta")
                    if oid in r.assigned_order_ids:
                        r.assigned_order_ids.remove(oid)
                # Replanificar: volver a la pizzería
                r.waypoints = [self.restaurant]
                r.waypoint_idx = 0
                path, _ = self.planner.astar(r.position, self.restaurant)
                r.route = path[1:] if path else []
                r.has_picked_up = False
                r.delivery_queue = []
                continue

            # --- lógica normal (plan + mover) ---
            if r.assigned_order_ids and not r.waypoints:
                self._rebuild_plan_for_rider(r)

            # --- 5.5) Batching: si está esperando en restaurante, no moverse todavía ---
            if r.wait_until > 0 and r.position == self.restaurant:
                if self.t < r.wait_until and r.can_take_more():
                    # Sigue esperando para agrupar más pedidos
                    continue
                else:
                    # Tiempo de espera terminado o ya tiene capacidad llena -> salir
                    r.wait_until = 0

            # mover 1 casilla
            if r.route:
                nxt = r.route.pop(0)
                r.position = nxt
                r.distance_travelled += 1.0

            tgt = r.current_target()
            if tgt is None:
                continue

            if r.position == tgt and r.waypoints:
                # (A) llega a restaurante y aún no ha recogido -> recoger
                if tgt == self.restaurant and (not r.has_picked_up) and r.assigned_order_ids:
                    r.has_picked_up = True

                    # ✅ marcar picked_up_at en todos sus pedidos pendientes
                    for oid in list(r.assigned_order_ids):
                        o = self._get_order(oid)
                        if o is not None and o.is_pending() and o.picked_up_at is None:
                            self.om.mark_picked_up(oid, now=self.t)

                    r.waypoint_idx += 1

                # (B) llega a drop sin haber recogido -> replan al restaurante
                elif tgt != self.restaurant and (not r.has_picked_up):
                    self._rebuild_plan_for_rider(r)

                # (C) llega a drop y ha recogido -> entregar primer drop pendiente
                elif tgt != self.restaurant and r.has_picked_up and r.delivery_queue:
                    oid = r.delivery_queue.pop(0)
                    o = self._get_order(oid)

                    if o is not None and o.is_pending():
                        self.om.mark_delivered(o.order_id, now=self.t)
                        delivered_now.append(o)
                        r.deliveries_done += 1

                    if oid in r.assigned_order_ids:
                        r.assigned_order_ids.remove(oid)

                    r.waypoint_idx += 1

                # (D) vuelve a restaurante y ya no quedan pedidos -> disponible
                elif tgt == self.restaurant and r.has_picked_up and (not r.assigned_order_ids):
                    r.available = True
                    r.has_picked_up = False
                    r.delivery_queue = []
                    r.waypoints = []
                    r.waypoint_idx = 0
                    r.route = []
                    continue

                nxt_tgt = r.current_target()
                if nxt_tgt is not None:
                    path, _ = self.planner.astar(r.position, nxt_tgt)
                    r.route = path[1:] if path else []
                else:
                    self._rebuild_plan_for_rider(r)

        return delivered_now

    # -------------------
    # State + Reward
    # -------------------
    def compute_state(self) -> tuple:
        pending = len(self.om.get_pending_orders())
        urgent = [o for o in self.om.get_pending_orders() if o.is_urgent(self.t) or o.priority > 1]
        urgent_ratio = (len(urgent) / pending) if pending > 0 else 0.0

        free = len([r for r in self.fm.get_all() if r.can_take_more()])

        deliveries = [r.deliveries_done for r in self.fm.get_all()]
        std_del = statistics.pstdev(deliveries) if len(deliveries) >= 2 else 0.0

        return make_state(
            t=self.t,
            episode_len=self.cfg.episode_len,
            pending=pending,
            urgent_ratio=urgent_ratio,
            free_riders=free,
            avg_fatigue=0.0,
            std_deliveries=std_del,
            traffic_level=self.traffic_level,
            closures=self.graph.count_closed_directed(),
        )

    def compute_reward(self, delivered_now: List[Order]) -> float:
        r = 0.0
        
        for o in delivered_now:
            if o.delivered_at <= o.deadline:
                # Bonus base
                r += 10.0
                # Bonus extra por urgente a tiempo
                if o.priority > 1:
                    r += 5.0
                # Bonus por entregar rápido (margen sobrante)
                margin = o.deadline - o.delivered_at
                r += min(3.0, margin * 0.1)  # Max +3 por rapidez
            else:
                late = o.delivered_at - o.deadline
                r -= (10.0 + 2.0 * late)
                # Penalización extra si era urgente
                if o.priority > 1:
                    r -= 5.0

        # Bonus por batching (entregar múltiples pedidos en un tick)
        if len(delivered_now) >= 2:
            r += 3.0 * (len(delivered_now) - 1)

        # Penalización por pedidos pendientes
        pending = self.om.get_pending_orders()
        r -= 0.2 * len(pending)
        
        # Penalización extra por pedidos muy viejos (>30 ticks esperando)
        for o in pending:
            age = self.t - o.created_at
            if age > 30:
                r -= 0.1 * (age - 30)
        
        return r

    # -------------------
    # Acción -> estrategia
    # -------------------
    def apply_action(self, action: int) -> None:
        all_orders = self.om.orders  # Todos los pedidos para calcular distancias

        if action == A_ASSIGN_URGENT_NEAREST:
            # Intentar asignar múltiples pedidos urgentes en un tick
            for _ in range(3):  # Máximo 3 asignaciones por tick
                orders = self.om.get_pending_orders()
                riders = self.fm.get_available_riders()
                pick = self.assigner.pick_urgent_nearest(orders, riders, now=self.t, all_orders=all_orders)
                if pick:
                    o, r = pick
                    self.assigner.assign(o, r)
                    self._rebuild_plan_for_rider(r)
                else:
                    break
            return

        if action == A_ASSIGN_ANY_NEAREST:
            # Intentar asignar múltiples pedidos en un tick (batching agresivo)
            for _ in range(3):  # Máximo 3 asignaciones por tick
                orders = self.om.get_pending_orders()
                riders = self.fm.get_available_riders()
                pick = self.assigner.pick_any_nearest(orders, riders, all_orders=all_orders)
                if pick:
                    o, r = pick
                    self.assigner.assign(o, r)
                    self._rebuild_plan_for_rider(r)
                else:
                    break
            return

        if action == A_WAIT:
            return

        if action == A_REPLAN_TRAFFIC:
            for r in self.fm.get_all():
                if r.waypoints:
                    self.assigner.replan_current_leg(r)
            return

    # -------------------
    # STEP
    # -------------------
    def step(self, action: int) -> tuple:
        self.maybe_change_traffic()
        self.maybe_road_closure()
        self.maybe_spawn_order()
        self.cancel_orders_in_blocked_zones()

        self.apply_action(action)

        delivered_now = self.move_riders_one_tick()
        reward = self.compute_reward(delivered_now)

        self.t += 1
        done = self.t >= self.cfg.episode_len

        if done:
            self.print_late_orders()

        return reward, done
