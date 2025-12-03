import heapq
import math

class RoutePlanner:
    def __init__(self, graph):
        self.graph = graph

    def heuristic(self, a, b):
        # Euclidiana para empezar (luego se puede mejorar)
        (x1, y1) = a
        (x2, y2) = b
        return math.dist((x1, y1), (x2, y2))

    def astar(self, start, goal):
        pq = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}

        while pq:
            _, current = heapq.heappop(pq)

            if current == goal:
                break

            for neighbor, cost in self.graph.edges.get(current, []):
                new_cost = cost_so_far[current] + cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(pq, (priority, neighbor))
                    came_from[neighbor] = current

        return came_from, cost_so_far
