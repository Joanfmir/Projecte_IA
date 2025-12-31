# core/route_planner.py
"""Planificación de rutas utilizando A*.

Este módulo implementa el algoritmo A* (A-Star) para encontrar caminos óptimos
en el grafo de la ciudad (`RoadGraph`), utilizando una heurística Octile/Chebyshev
adecuada para grids de 8 direcciones.
"""
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import heapq
from core.road_graph import RoadGraph, Node


class RoutePlanner:
    """Planificador de rutas utilizando A*.

    Attributes:
        graph: Instancia del grafo de la ciudad sobre el cual planificar.
    """

    def __init__(self, graph: RoadGraph):
        self.graph = graph

    @staticmethod
    def heuristic(a: Node, b: Node) -> float:
        """Heurística Octile: calcula distancia estimada óptima en grid 8-direccional.
        
        Coste diagonal ≈ 1.414, ortogonal = 1.0.
        Fórmula: max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)

        Args:
            a: Nodo origen.
            b: Nodo destino.

        Returns:
            Distancia estimada.
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + 0.414 * min(dx, dy)

    def astar(self, start: Node, goal: Node) -> Tuple[List[Node], float]:
        """Calcula el camino más corto entre start y goal usando A*.

        Args:
            start: Nodo de inicio.
            goal: Nodo destino.

        Returns:
            Una tupla (camino, coste_total).
            - camino: Lista de nodos desde start hasta goal.
            - coste_total: Coste acumulado del camino.
            Si no hay camino, retorna ([], inf).
        """
        # Cola de prioridad: (f_score, nodo)
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

        # Reconstruir camino
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path, cost_so_far[goal]
