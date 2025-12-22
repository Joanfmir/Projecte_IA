# core/order_manager.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

Node = Tuple[int, int]

@dataclass
class Order:
    order_id: int
    pickup: Node              # restaurante
    dropoff: Node             # cliente
    created_at: int
    deadline: int
    priority: int = 1

    assigned_to: Optional[int] = None
    delivered_at: Optional[int] = None

    def is_pending(self) -> bool:
        return self.delivered_at is None

    def is_urgent(self, now: int, threshold: int = 8) -> bool:
        return self.is_pending() and (self.deadline - now) <= threshold


class OrderManager:
    def __init__(self):
        self.orders: List[Order] = []
        self._next_id = 1

    def create_order(self, pickup: Node, dropoff: Node, now: int, max_eta: int, priority: int = 1) -> Order:
        o = Order(
            order_id=self._next_id,
            pickup=pickup,
            dropoff=dropoff,
            created_at=now,
            deadline=now + max_eta,
            priority=priority,
        )
        self._next_id += 1
        self.orders.append(o)
        return o

    def get_pending_orders(self) -> List[Order]:
        return [o for o in self.orders if o.is_pending()]

    def get_unassigned_pending(self) -> List[Order]:
        return [o for o in self.orders if o.is_pending() and o.assigned_to is None]

    def mark_delivered(self, order_id: int, now: int) -> None:
        for o in self.orders:
            if o.order_id == order_id:
                o.delivered_at = now
                return

    def stats_on_time(self) -> Tuple[int, int]:
        delivered = [o for o in self.orders if o.delivered_at is not None]
        if not delivered:
            return 0, 0
        on_time = sum(1 for o in delivered if o.delivered_at <= o.deadline)
        return on_time, len(delivered)
