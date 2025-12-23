# core/road_graph.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple
import random
import math

Node = Tuple[int, int]
Edge = Tuple[Node, Node]


@dataclass
class EdgeInfo:
    base_cost: float
    traffic_mult: float = 1.0
    closed: bool = False

    @property
    def cost(self) -> float:
        if self.closed:
            return float("inf")
        return self.base_cost * self.traffic_mult


class RoadGraph:
    """
    Grid graph con obstáculos (blocked=edificios) y movimientos 8-direcciones.
    Soporta:
      - cierres de calles (edges closed)
      - tráfico global (set_traffic_level)
      - tráfico por zonas (set_zone_traffic)
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
        self.nodes: Set[Node] = {
            (x, y)
            for x in range(width)
            for y in range(height)
            if (x, y) not in self.blocked
        }

        self.edges: Dict[Edge, EdgeInfo] = {}
        self._build_grid(base_cost)

        # tráfico actual (para snapshot/debug)
        self.global_traffic_level: str = "low"
        self.zone_levels: Dict[int, str] = {0: "low", 1: "low", 2: "low", 3: "low"}

    def _build_grid(self, base_cost: float) -> None:
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

    def neighbors(self, node: Node) -> Iterable[Tuple[Node, float]]:
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

            step_mult = math.sqrt(2) if (dx != 0 and dy != 0) else 1.0
            yield nb, base * step_mult

    # -------------------------
    # Tráfico
    # -------------------------
    @staticmethod
    def _mult_from_level(level: str) -> float:
        return {"low": 1.0, "medium": 1.5, "high": 2.2}.get(level, 1.0)

    def set_traffic_level(self, level: str) -> None:
        """Tráfico global (si lo quieres usar sin zonas)."""
        self.global_traffic_level = level
        mult = self._mult_from_level(level)
        for info in self.edges.values():
            if not info.closed:
                info.traffic_mult = mult

    def _zone_of_node(self, n: Node) -> int:
        """Zonas: 0 TL, 1 TR, 2 BL, 3 BR."""
        x, y = n
        midx = self.width / 2.0
        midy = self.height / 2.0
        left = x < midx
        bottom = y < midy
        if left and not bottom:
            return 0  # TL
        if (not left) and (not bottom):
            return 1  # TR
        if left and bottom:
            return 2  # BL
        return 3      # BR

    def set_zone_traffic(self, zone_levels: Dict[int, str]) -> None:
        """
        Tráfico por zonas: aplica un multiplicador diferente a cada arista según
        la zona donde cae (usamos el punto medio aproximado: el nodo origen).
        """
        # guarda niveles
        self.zone_levels = {0: "low", 1: "low", 2: "low", 3: "low"}
        for z, lvl in zone_levels.items():
            if z in self.zone_levels:
                self.zone_levels[z] = lvl

        # aplica a edges
        for (a, b), info in self.edges.items():
            if info.closed:
                continue
            z = self._zone_of_node(a)
            lvl = self.zone_levels.get(z, "low")
            info.traffic_mult = self._mult_from_level(lvl)

    # -------------------------
    # Cierres
    # -------------------------
    def random_road_incidents(self, n_closures: int) -> None:
        all_edges = list(self.edges.keys())
        self.rng.shuffle(all_edges)
        for (a, b) in all_edges[:n_closures]:
            self.close_edge(a, b)

    def close_edge(self, a: Node, b: Node) -> None:
        if (a, b) in self.edges:
            self.edges[(a, b)].closed = True

    def open_edge(self, a: Node, b: Node) -> None:
        if (a, b) in self.edges:
            self.edges[(a, b)].closed = False

    def count_closed_directed(self) -> int:
        return sum(1 for e in self.edges.values() if e.closed)

    def count_blocked(self) -> int:
        return len(self.blocked)

    def get_closed_edges(self) -> list[Edge]:
        return [(a, b) for (a, b), info in self.edges.items() if info.closed]
