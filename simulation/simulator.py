# simulation/simulator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List, Set
import random
import statistics

from core.road_graph import RoadGraph
from core.route_planner import RoutePlanner
from core.order_manager import OrderManager, Order
from core.fleet_manager import FleetManager, Rider
from core.assignment_engine import AssignmentEngine
from core.dispatch_policy import (
    make_state,
    A_ASSIGN_URGENT_NEAREST,
    A_ASSIGN_ANY_NEAREST,
    A_WAIT,
    A_REPLAN_TRAFFIC,
)

Node = Tuple[int, int]


@dataclass
class SimConfig:
    width: int = 25
    height: int = 25
    n_riders: int = 4
    episode_len: int = 300
    order_spawn_prob: float = 0.15
    max_eta: int = 55
    seed: int = 7

    # urban layout
    block_size: int = 5
    street_width: int = 1

    # cierres de calles
    road_closure_prob: float = 0.0
    road_closures_per_event: int = 1

    # Eventos internos separados (para control granular):
    # - enable_internal_spawn: generar pedidos cada tick (casi siempre True)
    # - enable_internal_traffic: cambiar tráfico cada 60 ticks (False si se maneja externamente)
    enable_internal_spawn: bool = True
    enable_internal_traffic: bool = True


class Simulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        self.avenues: List[dict] = []
        self.buildings: Set[Node] = self._generate_urban_buildings()

        self.graph = RoadGraph(
            cfg.width, cfg.height, base_cost=1.0, seed=cfg.seed, blocked=self.buildings
        )
        self.planner = RoutePlanner(self.graph)

        self.om = OrderManager()
        self.fm = FleetManager()

        self.restaurant: Node = self._nearest_walkable(
            (cfg.width // 2, cfg.height // 2)
        )
        self.assigner = AssignmentEngine(self.planner, restaurant_pos=self.restaurant)

        self.t = 0
        self.traffic_level = "low"
        # Tráfico por zonas (4 zonas: 0=NW, 1=NE, 2=SW, 3=SE)
        self.traffic_zones = {0: "low", 1: "low", 2: "low", 3: "low"}

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
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nb = (x + dx, y + dy)
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
        """
        Ordena los pedidos por EDF (Earliest Deadline First) con tie-break por distancia.
        Esto prioriza entregas urgentes sobre cercanas.
        """
        valid: List[Order] = []
        for oid in rider.assigned_order_ids:
            o = self._get_order(oid)
            if o is not None and o.is_pending():
                if o.dropoff == self.restaurant:
                    continue
                valid.append(o)

        if not valid:
            return []

        # EDF: ordenar por deadline primero, distancia como tie-break
        scored = []
        for o in valid:
            _, c = self.planner.astar(self.restaurant, o.dropoff)
            # (deadline, distancia) - deadline tiene prioridad
            scored.append((o.deadline, float(c), o.order_id))
        scored.sort(key=lambda x: (x[0], x[1]))  # deadline ASC, distancia ASC
        return [oid for _, _, oid in scored]

    def _rebuild_plan_for_rider(self, rider: Rider) -> None:
        rider.assigned_order_ids = [
            oid
            for oid in rider.assigned_order_ids
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
        pending_orders = [
            o for o in (self._get_order(oid) for oid in rider.assigned_order_ids) if o
        ]
        waypoints, dq = self.assigner.best_plan_for_rider(rider, pending_orders)

        rider.delivery_queue = dq
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

            orders_full.append(
                {
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
                }
            )

        return {
            "t": self.t,
            "restaurant": self.restaurant,
            "buildings": list(self.buildings),
            "avenues": list(self.avenues),
            "pending_orders": [
                (o.dropoff, o.priority, o.deadline, o.assigned_to) for o in pending
            ],
            "orders_full": orders_full,
            "riders": [
                {
                    "id": r.rider_id,
                    "pos": r.position,
                    "route": list(r.route),
                    "fatigue": r.fatigue,
                    "carrying": (
                        r.delivery_queue[0]
                        if (r.has_picked_up and r.delivery_queue)
                        else None
                    ),
                    "assigned": list(r.assigned_order_ids),
                    "picked": r.has_picked_up,
                    "waypoints": list(r.waypoints),
                    "wp_idx": r.waypoint_idx,
                    "available": r.available,
                    "resting": bool(getattr(r, "resting", False)),
                    # F7 FIX: Métricas para eval_factored.py
                    "distance": r.distance_travelled,
                    "deliveries_done": r.deliveries_done,
                }
                for r in riders
            ],
            "traffic": self.traffic_level,
            "traffic_zones": dict(self.traffic_zones),
            "closures": self.graph.count_closed_directed(),
            "blocked": self.graph.count_blocked(),
            "width": self.cfg.width,
            "height": self.cfg.height,
            "delivered_total": delivered_total,
            "delivered_ontime": delivered_ontime,
            "delivered_late": delivered_late,
        }

    # -------------------
    # Generación de pedidos / tráfico
    # -------------------
    def maybe_spawn_order(self) -> None:
        if self.rng.random() < self.cfg.order_spawn_prob:
            drop = self._random_walkable_cell()
            if drop is None:
                return
            prio = 2 if self.rng.random() < 0.15 else 1
            max_eta = self.cfg.max_eta if prio == 1 else int(self.cfg.max_eta * 0.6)

            self.om.create_order(
                pickup=self.restaurant,
                dropoff=drop,
                now=self.t,
                max_eta=max_eta,
                priority=prio,
            )

    def maybe_change_traffic(self) -> None:
        """Cambia el tráfico por zonas cada 60 ticks."""
        if self.t % 60 == 0 and self.t > 0:
            levels = ["low", "medium", "high"]
            # Cambiar cada zona con cierta probabilidad
            for zone in range(4):
                if self.rng.random() < 0.5:  # 50% de cambiar cada zona
                    self.traffic_zones[zone] = self.rng.choice(levels)

            # Actualizar el grafo con los nuevos niveles
            self.graph.set_zone_traffic(self.traffic_zones)

            # Mantener traffic_level global como el promedio/máximo
            pressure = {"low": 0, "medium": 1, "high": 2}
            avg = sum(pressure[lvl] for lvl in self.traffic_zones.values()) / 4
            if avg < 0.5:
                self.traffic_level = "low"
            elif avg < 1.5:
                self.traffic_level = "medium"
            else:
                self.traffic_level = "high"

    # -------------------
    # Movimiento + FATIGA
    # -------------------
    def move_riders_one_tick(self) -> Tuple[List[Order], int, float]:
        """
        Mueve riders un tick y procesa pickups/deliveries.

        Returns:
            Tuple[List[Order], int, float]: (pedidos entregados, número de pickups, distancia recorrida)
        """
        # --- parámetros fatiga (ajústalos aquí) ---
        FAT_STOP = 8.0  # si llega >= 8, se para
        FAT_RESUME = 6.0  # vuelve a moverse al bajar <= 6 (histeresis)
        FAT_DECAY = 0.02  # regenera siempre (si NO está descansando)
        FAT_DECAY_REST = 0.08  # regenera más rápido si está descansando
        FAT_MOVE_INC = 0.05  # sube por cada paso
        FAT_PICKUP_BONUS = 0.40  # sube extra al recoger pedidos

        delivered_now: List[Order] = []
        picked_up_count: int = 0  # Reward shaping: contador de pickups
        distance_moved: float = 0.0

        remaining_unassigned_global = [
            o for o in self.om.get_pending_orders() if o.assigned_to is None
        ]

        for r in self.fm.get_all():
            # --- 1) regeneración base ---
            resting = bool(getattr(r, "resting", False))
            if resting:
                r.fatigue = max(0.0, r.fatigue - FAT_DECAY_REST)
            else:
                r.fatigue = max(0.0, r.fatigue - FAT_DECAY)

            # --- 2) si estaba descansando, ¿ya puede volver? ---
            resting = bool(getattr(r, "resting", False))
            if resting and r.fatigue <= FAT_RESUME:
                setattr(r, "resting", False)
                resting = False

            # --- 3) si se pasa del límite, se pone a descansar y NO se mueve ---
            if (not resting) and r.fatigue >= FAT_STOP:
                setattr(r, "resting", True)
                r.available = False
                continue

            # --- 4) si está descansando, no se mueve ---
            if bool(getattr(r, "resting", False)):
                r.available = False
                continue

            # --- 5) lógica normal (plan + mover) ---
            if r.assigned_order_ids and not r.waypoints:
                self._rebuild_plan_for_rider(r)

            # mover 1 casilla
            if r.route:
                nxt = r.route.pop(0)
                r.position = nxt
                r.distance_travelled += 1.0
                distance_moved += 1.0
                r.fatigue += FAT_MOVE_INC  # ✅ sube al moverse

            tgt = r.current_target()
            if tgt is None:
                continue

            if r.position == tgt and r.waypoints:
                # (A) llega a restaurante y aún no ha recogido -> recoger
                if (
                    tgt == self.restaurant
                    and (not r.has_picked_up)
                    and r.assigned_order_ids
                ):
                    # Esperar en restaurante si queda backlog por asignar y hay capacidad
                    if remaining_unassigned_global and r.can_take_more():
                        r.available = False
                        continue

                    r.has_picked_up = True

                    # ✅ marcar picked_up_at en todos sus pedidos pendientes
                    for oid in list(r.assigned_order_ids):
                        o = self._get_order(oid)
                        if o is not None and o.is_pending() and o.picked_up_at is None:
                            self.om.mark_picked_up(oid, now=self.t)
                            picked_up_count += 1  # Reward shaping: contar pickups

                    # ✅ fatiga extra por el "trabajo" de recoger/cargar
                    r.fatigue += FAT_PICKUP_BONUS

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
                elif (
                    tgt == self.restaurant
                    and r.has_picked_up
                    and (not r.assigned_order_ids)
                ):
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

        return delivered_now, picked_up_count, distance_moved

    # -------------------
    # State + Reward
    # -------------------
    def compute_state(self) -> tuple:
        pending = len(self.om.get_pending_orders())
        urgent = [
            o
            for o in self.om.get_pending_orders()
            if o.is_urgent(self.t) or o.priority > 1
        ]
        urgent_ratio = (len(urgent) / pending) if pending > 0 else 0.0

        free = len([r for r in self.fm.get_all() if r.can_take_more()])

        fatigues = [r.fatigue for r in self.fm.get_all()]
        avg_fat = sum(fatigues) / len(fatigues) if fatigues else 0.0

        deliveries = [r.deliveries_done for r in self.fm.get_all()]
        std_del = statistics.pstdev(deliveries) if len(deliveries) >= 2 else 0.0

        return make_state(
            t=self.t,
            episode_len=self.cfg.episode_len,
            pending=pending,
            urgent_ratio=urgent_ratio,
            free_riders=free,
            avg_fatigue=avg_fat,
            std_deliveries=std_del,
            traffic_level=self.traffic_level,
            closures=self.graph.count_closed_directed(),
        )

    def compute_reward(
        self,
        delivered_now: List[Order],
        picked_up_count: int = 0,
        activation_count: int = 0,
        distance_moved: float = 0.0,
    ) -> float:
        """
        Calcula reward con shaping para pickups.

        Reward Shaping (CORREGIDO):
        - +3.0 por cada pickup (feedback inmediato aumentado)
        - +10.0 por entrega a tiempo
        - -10.0 -2*late por entrega tardía
        - -0.3 por pedido SIN ASIGNAR (no penaliza los que están en camino)
        - -0.02 * avg_fatigue (reducido para no dominar)
        - -activation_cost por cada rider que pasa de 0->1 pedidos (coste operativo)
        - -0.1 por cada unidad de distancia recorrida (eficiencia de ruta)
        """
        r = 0.0

        # Reward shaping: bonus por pickups (feedback inmediato) - AUMENTADO
        r += 3.0 * picked_up_count

        # Reward por entregas
        for o in delivered_now:
            if o.delivered_at <= o.deadline:
                r += 10.0
            else:
                late = o.delivered_at - o.deadline
                r -= 10.0 + 2.0 * late

        # CORRECCIÓN CRÍTICA: Solo penalizar pedidos SIN ASIGNAR
        # Antes: penalizaba TODOS los pendientes, incluso los que ya estaban en camino
        unassigned = [o for o in self.om.get_pending_orders() if o.assigned_to is None]
        r -= 0.5 * len(unassigned)

        # Coste de activar riders (batching friendly)
        r -= self.assigner.activation_cost * activation_count

        # Fatigue penalty REDUCIDO para no dominar el reward
        avg_fat = sum(x.fatigue for x in self.fm.get_all()) / max(
            1, len(self.fm.get_all())
        )
        r -= 0.02 * avg_fat

        # Coste por movimiento (estimula rutas compactas / batching)
        r -= 0.1 * distance_moved
        return r

    # -------------------
    # Acción -> estrategia (CON BATCHING)
    # -------------------
    def apply_action(self, action: int) -> Tuple[int, int]:
        """
        Aplica la acción seleccionada.
        BATCHING: asigna SOLO un par (pedido, rider) por tick.

        Returns:
            Tuple[int, int]: (#asignaciones, #activaciones_nuevas)
        """
        assigned_count = 0
        activation_count = 0

        if action == A_ASSIGN_URGENT_NEAREST:
            orders = self.om.get_pending_orders()
            riders = self.fm.get_all()
            pick = self.assigner.pick_urgent_nearest(orders, riders, now=self.t)
            if pick:
                o, r = pick
                was_empty = len(r.assigned_order_ids) == 0
                self.assigner.assign(o, r)
                self._rebuild_plan_for_rider(r)
                assigned_count = 1
                activation_count = 1 if was_empty else 0
            return assigned_count, activation_count

        if action == A_ASSIGN_ANY_NEAREST:
            orders = self.om.get_pending_orders()
            riders = self.fm.get_all()
            pick = self.assigner.pick_any_nearest(orders, riders, now=self.t)
            if pick:
                o, r = pick
                was_empty = len(r.assigned_order_ids) == 0
                self.assigner.assign(o, r)
                self._rebuild_plan_for_rider(r)
                assigned_count = 1
                activation_count = 1 if was_empty else 0
            return assigned_count, activation_count

        if action == A_WAIT:
            return 0, 0

        if action == A_REPLAN_TRAFFIC:
            for r in self.fm.get_all():
                if r.waypoints:
                    self.assigner.replan_current_leg(r)
            return 0, 0

        return assigned_count, activation_count

    # -------------------
    # STEP (MARKOVIANO)
    # -------------------
    def step(self, action: int) -> tuple:
        """
        Orden de eventos Markoviano:
        1. Observar estado (snapshot externo)
        2. Aplicar acción
        3. Mover riders (con tracking de pickups)
        4. Calcular reward (con shaping para pickups)
        5. Eventos aleatorios (afectan al SIGUIENTE estado, no al actual)
        6. Incrementar tiempo
        """
        # 1-2. Aplicar la acción elegida por el agente
        _, activation_count = self.apply_action(action)

        # 3. Mover riders y procesar entregas + pickups
        delivered_now, picked_up_count, distance_moved = self.move_riders_one_tick()

        # 4. Calcular reward con shaping (pickups dan +2.0 cada uno)
        reward = self.compute_reward(
            delivered_now,
            picked_up_count,
            activation_count=activation_count,
            distance_moved=distance_moved,
        )

        # 5. Eventos aleatorios (DESPUÉS de la acción, afectan siguiente snapshot)
        if self.cfg.enable_internal_traffic:
            self.maybe_change_traffic()
        if self.cfg.enable_internal_spawn:
            self.maybe_spawn_order()

        # 6. Avanzar tiempo
        self.t += 1
        done = self.t >= self.cfg.episode_len
        return reward, done
