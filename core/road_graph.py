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

            # Para movimiento diagonal, verificar que no haya obstáculo en el camino
            # No puedes pasar en diagonal si hay un edificio/bloqueo en las casillas adyacentes
            if dx != 0 and dy != 0:
                # Diagonal: verificar las dos casillas ortogonales
                side1 = (x + dx, y)  # horizontal
                side2 = (x, y + dy)  # vertical
                # Si alguna de las dos está bloqueada, no se puede pasar en diagonal
                if side1 not in self.nodes or side2 not in self.nodes:
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
        """
        Crea zonas bloqueadas que tienen sentido:
        - Obras: rectángulo 2x3 o 3x2 junto a un edificio
        - Accidente: línea de 3-4 casillas en una calle
        """
        for _ in range(n_closures):
            # Elegir tipo de incidente
            incident_type = self.rng.choice(["obras", "accidente"])
            
            if incident_type == "obras":
                self._create_construction_zone()
            else:
                self._create_accident_zone()

    def _create_construction_zone(self) -> None:
        """Crea una zona de obras rectangular junto a un edificio."""
        # Buscar una esquina de edificio
        corner = self._find_building_corner()
        if corner is None:
            return
        
        x, y = corner
        
        # Tamaños posibles de obras: 2x2, 2x3, 3x2
        sizes = [(2, 2), (2, 3), (3, 2)]
        self.rng.shuffle(sizes)
        
        for width, height in sizes:
            # Probar las 4 direcciones desde la esquina
            offsets = [(0, 0), (-width+1, 0), (0, -height+1), (-width+1, -height+1)]
            self.rng.shuffle(offsets)
            
            for ox, oy in offsets:
                nodes_to_block = []
                valid = True
                
                for dx in range(width):
                    for dy in range(height):
                        node = (x + ox + dx, y + oy + dy)
                        if node not in self.nodes:
                            valid = False
                            break
                        # Verificar que no bloquee un pasillo único
                        if not self._can_safely_block(node):
                            valid = False
                            break
                        nodes_to_block.append(node)
                    if not valid:
                        break
                
                if valid and len(nodes_to_block) >= 3:
                    for node in nodes_to_block:
                        self._block_node(node)
                    return

    def _create_accident_zone(self) -> None:
        """Crea una banda de bloqueo en una calle (accidente/avería), más ancha."""
        start = self._find_street_node()
        if start is None:
            return

        # Elegir dirección (horizontal o vertical)
        directions = [(1, 0), (0, 1)]
        self.rng.shuffle(directions)

        for dx, dy in directions:
            nodes_to_block = []
            current = start
            length = self.rng.randint(3, 5)
            width = self.rng.choice([2, 3])  # ancho de la banda
            for _ in range(length):
                # Para cada posición en la banda, bloquear celdas adyacentes
                for w in range(-width//2, width//2 + 1):
                    if dx == 1:  # horizontal
                        node = (current[0], current[1] + w)
                    else:       # vertical
                        node = (current[0] + w, current[1])
                    if node in self.nodes and self._can_safely_block(node):
                        nodes_to_block.append(node)
                current = (current[0] + dx, current[1] + dy)
                if current not in self.nodes:
                    break
            # Filtrar duplicados
            nodes_to_block = list(set(nodes_to_block))
            if len(nodes_to_block) >= 6:
                for node in nodes_to_block:
                    self._block_node(node)
                return

    def _find_building_corner(self) -> Optional[Node]:
        """Encuentra un nodo que esté en una esquina de edificio."""
        candidates = []
        for node in list(self.nodes):
            x, y = node
            # Contar edificios adyacentes (solo ortogonales)
            building_dirs = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if (x + dx, y + dy) in self.blocked:
                    building_dirs.append((dx, dy))
            
            # Esquina: tiene edificio en 2 direcciones perpendiculares
            if len(building_dirs) >= 2:
                candidates.append(node)
        
        if not candidates:
            return None
        return self.rng.choice(candidates)

    def _find_street_node(self) -> Optional[Node]:
        """Encuentra un nodo en medio de una calle (con vecinos en línea)."""
        candidates = []
        for node in list(self.nodes):
            x, y = node
            # Buscar nodos con vecinos alineados (calle)
            h_neighbors = sum(1 for dx in [-1, 1] if (x + dx, y) in self.nodes)
            v_neighbors = sum(1 for dy in [-1, 1] if (x, y + dy) in self.nodes)
            
            # Es calle si tiene 2 vecinos en línea (horizontal o vertical)
            if h_neighbors == 2 or v_neighbors == 2:
                candidates.append(node)
        
        if not candidates:
            return None
        return self.rng.choice(candidates)

    def _can_safely_block(self, node: Node) -> bool:
        """Verifica que bloquear este nodo no deje zonas inaccesibles."""
        if node not in self.nodes:
            return False
        
        x, y = node
        walkable_neighbors = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (x + dx, y + dy) in self.nodes:
                walkable_neighbors += 1
        
        # Solo bloquear si hay suficientes alternativas
        return walkable_neighbors >= 2

    def _block_node(self, node: Node) -> None:
        """Bloquea un nodo convirtiéndolo en edificio temporal."""
        if node not in self.nodes:
            return
        
        self.nodes.discard(node)
        self.blocked.add(node)
        
        # Cerrar todas las aristas de este nodo
        for edge in list(self.edges.keys()):
            if node in edge:
                self.edges[edge].closed = True

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

    def get_closed_edges_sample(self, max_edges: int = 200) -> List[Edge]:
        """Para el visual: devuelve una muestra de aristas cerradas (dirigidas)."""
        out: List[Edge] = []
        for (a, b), info in self.edges.items():
            if info.closed:
                out.append((a, b))
                if len(out) >= max_edges:
                    break
        return out
