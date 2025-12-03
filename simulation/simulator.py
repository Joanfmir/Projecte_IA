class Simulator:
    def __init__(self, order_manager, fleet_manager, assignment_engine):
        self.om = order_manager
        self.fm = fleet_manager
        self.assigner = assignment_engine

    def tick(self):
        orders = self.om.get_pending_orders()
        riders = self.fm.get_available_riders()
        assignments = self.assigner.assign(orders, riders)
        return assignments
