# simulation/simulator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List, Set
import random
import statistics

from core.road_graph import RoadGraph
from core.route_planner import RoutePlanner
from core.order_manager import OrderManager, Order
from core.fleet_manager import FleetManager
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
    episode_len: int = 400
    order_spawn_prob: float = 0.35
    max_eta: int = 55
    seed: int = 7

    # urban layout
    block_size: int = 5      # tamaño manzana (edificio)
    street_width: int = 1    # ancho calles


class Simulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        # Avenidas (para el visual)
        self.avenues: List[dict] = []

        # 1) Generar ciudad: buildings (obstáculos) + avenues (decoración recta)
        self.buildings: Set[Node] = self._generate_urban_buildings()

        # 2) Grafo con obstáculos
        self.graph = RoadGraph(cfg.width, cfg.height, base_cost=1.0, seed=cfg.seed, blocked=self.buildings)
        self.planner = RoutePlanner(self.graph)

        self.om = OrderManager()
        self.fm = FleetManager()

        # Restaurante: celda caminable cerca del centro
        self.restaurant: Node = self._nearest_walkable((cfg.width // 2, cfg.height // 2))
        self.assigner = AssignmentEngine(self.planner, restaurant_pos=self.restaurant)

        self.t = 0
        self.traffic_level = "low"

        for _ in range(cfg.n_riders):
            sp = self.rng.choice([0.9, 1.0, 1.1])
            self.fm.add_rider(position=self.restaurant, speed=sp)

    def reset(self) -> None:
        self.__init__(self.cfg)

    # -----------------------
    # URBAN GENERATION
    # -----------------------
    def _generate_urban_buildings(self) -> Set[Node]:
        """
        Ciudad:
        - Manzanas sólidas (sin agujeros)
        - Calles ortogonales (block_size + street_width)
        - 1-3 avenidas diagonales rectas random
        """
        W, H = self.cfg.width, self.cfg.height
        bs = self.cfg.block_size
        sw = self.cfg.street_width

        buildings: Set[Node] = set()
        self.avenues = []

        # 1) Manzanas sólidas
        step = bs + sw
        for bx in range(0, W, step):
            for by in range(0, H, step):
                for x in range(bx, min(bx + bs, W)):
                    for y in range(by, min(by + bs, H)):
                        buildings.add((x, y))

        # 2) Borde exterior siempre calle
        for x in range(W):
            buildings.discard((x, 0))
            buildings.discard((x, H - 1))
        for y in range(H):
            buildings.discard((0, y))
            buildings.discard((W - 1, y))

        # 3) Avenidas diagonales random (la lógica las abre como “celdas calle”)
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

            # Guardar para dibujar recto en el visualizer (solo visual)
            self.avenues.append({"m": m, "b": b, "w": avenue_width})

            # Abrir calle siguiendo la recta (en grid)
            for x in range(W):
                y_center = m * x + b
                y_round = int(round(y_center))
                for dy in range(-avenue_width, avenue_width + 1):
                    y = y_round + dy
                    if 0 <= y < H:
                        buildings.discard((x, y))

            # continuidad vertical para pendientes grandes
            if abs(m) > 0.01:
                for y in range(H):
                    x_center = (y - b) / m
                    x_round = int(round(x_center))
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

    def _random_walkable_cell(self) -> Node:
        for _ in range(5000):
            x = self.rng.randrange(self.cfg.width)
            y = self.rng.randrange(self.cfg.height)
            if self.graph.is_walkable((x, y)):
                return (x, y)
        return self.restaurant

    # -------------------
    # VISUAL / SNAPSHOT
    # -------------------
    def snapshot(self) -> dict:
        pending = self.om.get_pending_orders()
        riders = self.fm.get_all()

        # métricas delivered (para train/eval)
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

        return {
            "t": self.t,
            "restaurant": self.restaurant,
            "buildings": list(self.buildings),
            "avenues": list(self.avenues),
            "pending_orders": [(o.dropoff, o.priority, o.deadline, o.assigned_to) for o in pending],
            "riders": [
                {
                    "id": r.rider_id,
                    "pos": r.position,
                    "route": list(r.route),
                    "fatigue": r.fatigue,
                    "carrying": (r.assigned_order_id if r.has_picked_up else None),
                }
                for r in riders
            ],
            "traffic": self.traffic_level,
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
            prio = 2 if self.rng.random() < 0.15 else 1
            max_eta = self.cfg.max_eta if prio == 1 else int(self.cfg.max_eta * 0.6)

            self.om.create_order(
                pickup=self.restaurant,
                dropoff=drop,
                now=self.t,
                max_eta=max_eta,
                priority=prio
            )

    def maybe_change_traffic(self) -> None:
        if self.t % 60 == 0 and self.t > 0:
            self.traffic_level = self.rng.choice(["low", "medium", "high"])
            self.graph.set_traffic_level(self.traffic_level)

    # -------------------
    # Helpers orders
    # -------------------
    def _get_order(self, order_id: int) -> Optional[Order]:
        for o in self.om.orders:
            if o.order_id == order_id:
                return o
        return None

    # -------------------
    # Movimiento (pickup/dropoff/return)
    # -------------------
    def move_riders_one_tick(self) -> List[Order]:
        delivered_now: List[Order] = []

        for r in self.fm.get_all():
            if r.route:
                nxt = r.route.pop(0)
                r.position = nxt
                r.distance_travelled += 1.0
                r.fatigue += 0.05

            target = r.current_target()
            if target is not None and r.position == target and r.has_task():
                o = self._get_order(r.assigned_order_id)

                if r.waypoint_idx == 0:
                    r.has_picked_up = True
                    r.waypoint_idx = 1

                elif r.waypoint_idx == 1:
                    if o is not None and o.is_pending():
                        self.om.mark_delivered(o.order_id, now=self.t)
                        delivered_now.append(o)
                        r.deliveries_done += 1
                    r.waypoint_idx = 2

                elif r.waypoint_idx == 2:
                    r.assigned_order_id = None
                    r.has_picked_up = False
                    r.waypoints = []
                    r.waypoint_idx = 0
                    r.route = []
                    r.available = True
                    continue

                if r.has_task():
                    new_target = r.current_target()
                    if new_target is not None:
                        path, _ = self.planner.astar(r.position, new_target)
                        r.route = path[1:] if path else []

        return delivered_now

    # -------------------
    # State + Reward
    # -------------------
    def compute_state(self) -> tuple:
        pending = len(self.om.get_pending_orders())
        urgent = [o for o in self.om.get_pending_orders() if o.is_urgent(self.t) or o.priority > 1]
        urgent_ratio = (len(urgent) / pending) if pending > 0 else 0.0

        free = len(self.fm.get_available_riders())
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

    def compute_reward(self, delivered_now: List[Order]) -> float:
        r = 0.0
        for o in delivered_now:
            if o.delivered_at <= o.deadline:
                r += 10.0
            else:
                late = o.delivered_at - o.deadline
                r -= (10.0 + 2.0 * late)

        r -= 0.2 * len(self.om.get_pending_orders())
        avg_fat = sum(x.fatigue for x in self.fm.get_all()) / max(1, len(self.fm.get_all()))
        r -= 0.05 * avg_fat
        return r

    # -------------------
    # Acción -> estrategia
    # -------------------
    def apply_action(self, action: int) -> None:
        orders = self.om.get_pending_orders()
        riders = self.fm.get_available_riders()

        if action == A_ASSIGN_URGENT_NEAREST:
            pick = self.assigner.pick_urgent_nearest(orders, riders, now=self.t)
            if pick:
                o, r = pick
                self.assigner.assign(o, r)
            return

        if action == A_ASSIGN_ANY_NEAREST:
            pick = self.assigner.pick_any_nearest(orders, riders)
            if pick:
                o, r = pick
                self.assigner.assign(o, r)
            return

        if action == A_WAIT:
            return

        if action == A_REPLAN_TRAFFIC:
            for r in self.fm.get_all():
                if r.has_task():
                    self.assigner.replan_current_leg(r)
            return

    # -------------------
    # STEP
    # -------------------
    def step(self, action: int) -> tuple:
        self.maybe_change_traffic()
        self.maybe_spawn_order()

        self.apply_action(action)

        delivered_now = self.move_riders_one_tick()
        reward = self.compute_reward(delivered_now)

        self.t += 1
        done = self.t >= self.cfg.episode_len
        return reward, done
