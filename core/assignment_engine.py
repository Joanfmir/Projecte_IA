class AssignmentEngine:
    def __init__(self, route_planner):
        self.route_planner = route_planner

    def assign(self, orders, riders):
        assignments = []
        for order in orders:
            best = None
            best_cost = float('inf')
            for r in riders:
                _, cost = self.route_planner.astar(r.position, order.location)
                if cost[order.location] < best_cost:
                    best = r
                    best_cost = cost[order.location]
            assignments.append((order, best))
        return assignments

