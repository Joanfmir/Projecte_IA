# core/order_manager.py
"""Gestión de pedidos (Orders).

Define la estructura de datos para un pedido y la clase gestora encargada de su
ciclo de vida (creación, asignación, recolección y entrega).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

Node = Tuple[int, int]


@dataclass
class Order:
    """Representa un pedido en el sistema.

    Attributes:
        order_id: Identificador único del pedido.
        pickup: Coordenadas de recogida (restaurante).
        dropoff: Coordenadas de entrega (cliente).
        created_at: Tick de simulación en que se creó el pedido.
        deadline: Tick límite para la entrega.
        priority: Prioridad del pedido (1=normal, >1=urgente).
        assigned_to: ID del rider asignado (None si no está asignado).
        picked_up_at: Tick en que fue recogido por el rider.
        delivered_at: Tick en que fue entregado al cliente.
    """
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
        """Determina si el pedido aún no ha sido entregado."""
        return self.delivered_at is None

    def is_picked(self) -> bool:
        """Determina si el pedido ya fue recogido pero no entregado (en tránsito)."""
        return self.picked_up_at is not None and self.delivered_at is None

    def is_urgent(self, now: int) -> bool:
        """Verifica si el pedido se considera urgente.

        Un pedido es urgente si tiene prioridad alta (>1) o si está cerca
        de su deadline (menos del 25% del tiempo inicial restante).

        Args:
            now: Tick actual de la simulación.

        Returns:
            True si es urgente, False en caso contrario.
        """
        if self.priority > 1:
            return True
        remaining = self.deadline - now
        total_window = max(1, self.deadline - self.created_at)
        return remaining <= 0.25 * total_window


class OrderManager:
    """Gestor centralizado de pedidos.

    Se encarga de crear nuevos pedidos, buscar pedidos y actualizar sus estados.

    Attributes:
        orders: Lista de todos los pedidos históricos y activos.
        _next_id: Contador para IDs únicos.
    """

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
        """Crea un nuevo pedido y lo registra.

        Args:
            pickup: Ubicación de recogida.
            dropoff: Ubicación de entrega.
            now: Tiempo de creación.
            max_eta: Tiempo máximo permitido para la entrega (para calcular deadline).
            priority: Nivel de prioridad.

        Returns:
            La instancia del Order creado.
        """
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
        """Busca un pedido por su ID."""
        for o in self.orders:
            if o.order_id == order_id:
                return o
        return None

    def get_pending_orders(self) -> List[Order]:
        """Obtiene una lista de todos los pedidos no entregados."""
        return [o for o in self.orders if o.delivered_at is None]

    def mark_assigned(self, order_id: int, rider_id: int) -> None:
        """Marca un pedido como asignado a un rider específico."""
        o = self.get_order(order_id)
        if o is None:
            return
        o.assigned_to = rider_id

    def mark_picked_up(self, order_id: int, now: int) -> None:
        """Marca un pedido como recogido por su rider asignado."""
        o = self.get_order(order_id)
        if o is None:
            return
        if o.picked_up_at is None:
            o.picked_up_at = now

    def mark_delivered(self, order_id: int, now: int) -> None:
        """Marca un pedido como entregado (finalizado)."""
        o = self.get_order(order_id)
        if o is None:
            return
        o.delivered_at = now
