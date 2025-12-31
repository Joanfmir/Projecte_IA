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

        # para visual/depuración (opcional)
        self.last_zone_levels: Optional[Dict[int, str]] = None

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

    # -------- tráfico y cierres --------
    def set_traffic_level(self, level: str) -> None:
        """Tráfico global (si lo usas)."""
        mult = {"low": 1.0, "medium": 1.5, "high": 2.2}.get(level, 1.0)
        for info in self.edges.values():
            if not info.closed:
                info.traffic_mult = mult
        self.last_zone_levels = None

    def _zone_of(self, node: Node) -> int:
        """4 zonas por cuadrantes: 0 TL, 1 TR, 2 BL, 3 BR."""
        x, y = node
        midx = self.width // 2
        midy = self.height // 2
        left = x < midx
        top = y >= midy
        if left and top:
            return 0  # TL
        if (not left) and top:
            return 1  # TR
        if left and (not top):
            return 2  # BL
        return 3      # BR

    def set_zone_traffic(self, zone_levels: Dict[int, str]) -> None:
        """
        Tráfico por zonas (cuadrantes).
        zone_levels: {0:"high",1:"medium",2:"medium",3:"low"} por ejemplo.
        El multiplicador se aplica según la zona del nodo ORIGEN de la arista.
        """
        mult_map = {"low": 1.0, "medium": 1.5, "high": 2.2}
        # guardo por si quieres verlo en snapshot
        self.last_zone_levels = dict(zone_levels)

        for (a, b), info in self.edges.items():
            if info.closed:
                continue
            z = self._zone_of(a)
            lvl = zone_levels.get(z, "low")
            info.traffic_mult = mult_map.get(lvl, 1.0)

    def random_road_incidents(self, n_closures: int) -> None:
        all_edges = list(self.edges.keys())
        self.rng.shuffle(all_edges)
        for (a, b) in all_edges[:n_closures]:
            self.close_edge(a, b)

    def close_edge(self, a: Node, b: Node) -> None:
        """Cierra la arista en ambas direcciones (bidireccional)."""
        if (a, b) in self.edges:
            self.edges[(a, b)].closed = True
        if (b, a) in self.edges:
            self.edges[(b, a)].closed = True

    def open_edge(self, a: Node, b: Node) -> None:
        if (a, b) in self.edges:
            self.edges[(a, b)].closed = False

    def count_closed_directed(self) -> int:
        return sum(1 for e in self.edges.values() if e.closed)

    def count_blocked(self) -> int:
        return len(self.blocked)

    def get_closed_edges_sample(self, max_edges: int = 200) -> List[Edge]:
        """Para el visual: devuelve una muestra de aristas cerradas (dirigidas)."""
        out: List[Edge] = []
        for (a, b), info in self.edges.items():
            if info.closed:
                out.append((a, b))
                if len(out) >= max_edges:
                    break
        return out
