# core/route_planner.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import heapq
from core.road_graph import RoadGraph, Node

class RoutePlanner:
    def __init__(self, graph: RoadGraph):
        self.graph = graph

    @staticmethod
    def heuristic(a: Node, b: Node) -> float:
        """Octile distance: óptima para grids 8-direccionales."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + 0.414 * min(dx, dy)

    def astar(self, start: Node, goal: Node) -> Tuple[List[Node], float]:
        """
        Devuelve (camino, coste_total). Si no hay camino, ([], inf)
        """
        pq: List[Tuple[float, Node]] = []
        heapq.heappush(pq, (0.0, start))

        came_from: Dict[Node, Optional[Node]] = {start: None}
        cost_so_far: Dict[Node, float] = {start: 0.0}

        while pq:
            _, current = heapq.heappop(pq)
            if current == goal:
                break

            for nb, edge_cost in self.graph.neighbors(current):
                new_cost = cost_so_far[current] + edge_cost
                if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                    cost_so_far[nb] = new_cost
                    priority = new_cost + self.heuristic(nb, goal)
                    heapq.heappush(pq, (priority, nb))
                    came_from[nb] = current

        if goal not in came_from:
            return [], float("inf")

        # reconstruir camino
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path, cost_so_far[goal]
