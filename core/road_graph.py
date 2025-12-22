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
    Grid graph con obstáculos (blocked=edificios) y movimientos 8-direcciones (diagonales).
    - Si una celda está en blocked: NO se puede pisar.
    - Diagonal tiene coste sqrt(2) respecto a ortogonal.
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

            # coste diagonal sqrt(2)
            step_mult = math.sqrt(2) if (dx != 0 and dy != 0) else 1.0
            yield nb, base * step_mult

    # -------- tráfico y cierres --------
    def set_traffic_level(self, level: str) -> None:
        mult = {"low": 1.0, "medium": 1.5, "high": 2.2}.get(level, 1.0)
        for info in self.edges.values():
            if not info.closed:
                info.traffic_mult = mult

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
