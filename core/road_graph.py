class RoadGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}  # {node: [(neighbor, cost), ...]}

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, a, b, cost):
        self.edges.setdefault(a, []).append((b, cost))
        self.edges.setdefault(b, []).append((a, cost))  # si es bidireccional
