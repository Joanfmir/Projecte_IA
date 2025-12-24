# core/order_manager.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

Node = Tuple[int, int]


@dataclass
class Order:
    order_id: int
    pickup: Node
    dropoff: Node
    created_at: int
    deadline: int
    priority: int = 1
    assigned_to: Optional[int] = None

    # ✅ NUEVO: momento de recogida
    picked_up_at: Optional[int] = None

    delivered_at: Optional[int] = None

    def is_pending(self) -> bool:
        return self.delivered_at is None

    def is_picked(self) -> bool:
        return self.picked_up_at is not None and self.delivered_at is None

    def is_urgent(self, now: int) -> bool:
        if self.priority > 1:
            return True
        remaining = self.deadline - now
        total_window = max(1, self.deadline - self.created_at)
        return remaining <= 0.25 * total_window


class OrderManager:
    def __init__(self):
        self.orders: List[Order] = []
        self._next_id = 1

    def create_order(
        self,
        pickup: Node,
        dropoff: Node,
        now: int,
        max_eta: int,
        priority: int = 1
    ) -> Order:
        oid = self._next_id
        self._next_id += 1

        deadline = now + max_eta

        o = Order(
            order_id=oid,
            pickup=pickup,
            dropoff=dropoff,
            created_at=now,
            deadline=deadline,
            priority=priority,
            assigned_to=None,
            picked_up_at=None,
            delivered_at=None
        )
        self.orders.append(o)
        return o

    def get_order(self, order_id: int) -> Optional[Order]:
        for o in self.orders:
            if o.order_id == order_id:
                return o
        return None

    def get_pending_orders(self) -> List[Order]:
        return [o for o in self.orders if o.delivered_at is None]

    def mark_assigned(self, order_id: int, rider_id: int) -> None:
        o = self.get_order(order_id)
        if o is None:
            return
        o.assigned_to = rider_id

    # ✅ NUEVO
    def mark_picked_up(self, order_id: int, now: int) -> None:
        o = self.get_order(order_id)
        if o is None:
            return
        if o.picked_up_at is None:
            o.picked_up_at = now

    def mark_delivered(self, order_id: int, now: int) -> None:
        o = self.get_order(order_id)
        if o is None:
            return
        o.delivered_at = now
