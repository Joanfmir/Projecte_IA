class Order:
    def __init__(self, id, location, time_limit, created_at):
        self.id = id
        self.location = location
        self.time_limit = time_limit
        self.created_at = created_at
        self.assigned_to = None

class OrderManager:
    def __init__(self):
        self.orders = []

    def add_order(self, order):
        self.orders.append(order)

    def get_pending_orders(self):
        return [o for o in self.orders if o.assigned_to is None]
