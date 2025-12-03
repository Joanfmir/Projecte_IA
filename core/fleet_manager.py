class Rider:
    def __init__(self, id, position, fatigue=0, speed=1.0):
        self.id = id
        self.position = position
        self.fatigue = fatigue
        self.speed = speed
        self.available = True

class FleetManager:
    def __init__(self):
        self.riders = []

    def add_rider(self, rider):
        self.riders.append(rider)

    def get_available_riders(self):
        return [r for r in self.riders if r.available]
