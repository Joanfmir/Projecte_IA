# core/road_graph.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple, List
import random
import math

Node = Tuple[int, int]
Edge = Tuple[Node, Node]


@dataclass
class EdgeInfo:
    base_cost: float
    traffic_mult: float = 1.0   # multiplicador global (si usas set_traffic_level)
    closed: bool = False

    @property
    def cost(self) -> float:
        if self.closed:
            return float("inf")
        return self.base_cost * self.traffic_mult


class RoadGraph:
    """
    Grid graph con obstáculos (blocked=edificios) y movimientos 8-direcciones (diagonales).
    - Si una celda está en blocked: NO se puede pisar.
    - Diagonal tiene coste sqrt(2) respecto a ortogonal.

    EXTENSIONES:
    - Tráfico por zonas (4 cuadrantes): factor multiplicador adicional por zona.
    - Cierres temporales (TTL) de calles/aristas: se cierran y se reabren solos con tick_closures().
    """

    def __init__(
        self,
        width: int,
        height: int,
        base_cost: float = 1.0,
        seed: int = 42,
        blocked: Optional[Set[Node]] = None,
    ):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)

        self.blocked: Set[Node] = set(blocked) if blocked else set()

        # nodos transitables
        self.nodes: Set[Node] = {
            (x, y)
            for x in range(width)
            for y in range(height)
            if (x, y) not in self.blocked
        }

        # aristas dirigidas
        self.edges: Dict[Edge, EdgeInfo] = {}
        self._build_grid(base_cost)

        # ----------------------------
        # NUEVO: tráfico por zonas (4 cuadrantes)
        # ----------------------------
        # 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
        self.zone_levels: Dict[int, str] = {0: "low", 1: "medium", 2: "medium", 3: "high"}
        self.zone_factors: Dict[str, float] = {"low": 1.0, "medium": 1.35, "high": 1.8}

        # ----------------------------
        # NUEVO: cierres temporales por TTL
        # ----------------------------
        # guardamos TTL solo para las aristas cerradas por “obras”
        self.closed_edges_ttl: Dict[Edge, int] = {}

    def _build_grid(self, base_cost: float) -> None:
        # Creamos aristas para 8 direcciones, solo entre nodos existentes
        dirs = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]
        for (x, y) in self.nodes:
            a = (x, y)
            for dx, dy in dirs:
                b = (x + dx, y + dy)
                if b in self.nodes:
                    self.edges[(a, b)] = EdgeInfo(base_cost=base_cost)

    def is_walkable(self, node: Node) -> bool:
        return node in self.nodes

    # ----------------------------
    # NUEVO: zonas de tráfico
    # ----------------------------
    def zone_id(self, node: Node) -> int:
        x, y = node
        midx = self.width // 2
        midy = self.height // 2
        # 0 TL, 1 TR, 2 BL, 3 BR (tomamos "top" como y >= midy)
        top = y >= midy
        left = x < midx
        if top and left:
            return 0
        if top and not left:
            return 1
        if (not top) and left:
            return 2
        return 3

    def set_zone_traffic(self, zone_levels: Dict[int, str]) -> None:
        """
        zone_levels ejemplo: {0:"low",1:"high",2:"medium",3:"low"}
        """
        for z, lvl in zone_levels.items():
            if z in (0, 1, 2, 3):
                self.zone_levels[z] = lvl

    def traffic_factor_for_edge(self, u: Node, v: Node) -> float:
        # factor según zona del origen u (simple y estable)
        zid = self.zone_id(u)
        lvl = self.zone_levels.get(zid, "low")
        return self.zone_factors.get(lvl, 1.0)

    # ----------------------------
    # Vecinos + costes
    # ----------------------------
    def neighbors(self, node: Node) -> Iterable[Tuple[Node, float]]:
        # OJO: 8 direcciones
        dirs = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]

        x, y = node
        for dx, dy in dirs:
            nb = (x + dx, y + dy)
            if nb not in self.nodes:
                continue

            info = self.edges.get((node, nb))
            if info is None:
                continue

            base = info.cost
            if base == float("inf"):
                continue

            # factor de zona (cuadrante)
            base *= self.traffic_factor_for_edge(node, nb)

            # coste diagonal sqrt(2)
            step_mult = math.sqrt(2) if (dx != 0 and dy != 0) else 1.0
            yield nb, base * step_mult

    # -------- tráfico global (se mantiene) --------
    def set_traffic_level(self, level: str) -> None:
        """
        Mantengo esto por compatibilidad.
        Ajusta un multiplicador global en TODAS las aristas.
        """
        mult = {"low": 1.0, "medium": 1.5, "high": 2.2}.get(level, 1.0)
        for info in self.edges.values():
            if not info.closed:
                info.traffic_mult = mult

    # -------- cierres (TTL) --------
    def random_road_incidents(self, n_closures: int, ttl: int = 80) -> None:
        """
        Cierra n_closures aristas aleatorias con un TTL por defecto.
        """
        all_edges = list(self.edges.keys())
        self.rng.shuffle(all_edges)
        for (a, b) in all_edges[:n_closures]:
            self.close_edge(a, b, ttl=ttl)

    def close_edge(self, a: Node, b: Node, ttl: Optional[int] = None) -> None:
        """
        Cierra la arista dirigida (a->b).
        Si ttl es None: se cierra "permanente" (hasta open_edge).
        Si ttl es int: se cierra durante ttl ticks y luego se reabre solo con tick_closures().
        """
        e = (a, b)
        if e in self.edges:
            self.edges[e].closed = True
            if ttl is not None:
                self.closed_edges_ttl[e] = max(1, int(ttl))

    def open_edge(self, a: Node, b: Node) -> None:
        e = (a, b)
        if e in self.edges:
            self.edges[e].closed = False
        # si estaba con TTL, lo quitamos
        if e in self.closed_edges_ttl:
            del self.closed_edges_ttl[e]

    def tick_closures(self) -> None:
        """
        Decrementa TTL y reabre cuando llega a 0.
        Llamar 1 vez por tick desde el Simulator.
        """
        if not self.closed_edges_ttl:
            return

        to_open: List[Edge] = []
        for e, t in list(self.closed_edges_ttl.items()):
            t2 = t - 1
            if t2 <= 0:
                to_open.append(e)
            else:
                self.closed_edges_ttl[e] = t2

        for (a, b) in to_open:
            self.open_edge(a, b)

    def get_closed_edges(self) -> List[Edge]:
        """
        Para visualización: devuelve lista de aristas cerradas activas (TTL).
        (Si quisieras incluir también cierres permanentes, lo ampliamos.)
        """
        return list(self.closed_edges_ttl.keys())

    # -------- contadores --------
    def count_closed_directed(self) -> int:
        return sum(1 for e in self.edges.values() if e.closed)

    def count_blocked(self) -> int:
        return len(self.blocked)
