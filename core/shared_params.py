# core/shared_params.py
"""Constantes de acción compartidas entre el agente factorizado y la política heurística.

Estas constantes definen el espacio de acciones del agente y son consumidas por
el simulador, los agentes y los tests.
"""
from __future__ import annotations

# Acciones disponibles
A_ASSIGN_URGENT_NEAREST = 0  # Asignar pedido urgente al rider más cercano.
A_ASSIGN_ANY_NEAREST = 1     # Asignar cualquier pedido pendiente al más cercano.
A_WAIT = 2                   # No hacer nada este tick.
A_REPLAN_TRAFFIC = 3         # Forzar replanificación por cambio de tráfico.

# Lista de todas las acciones
ALL_ACTIONS = [
    A_ASSIGN_URGENT_NEAREST,
    A_ASSIGN_ANY_NEAREST,
    A_WAIT,
    A_REPLAN_TRAFFIC,
]

__all__ = [
    "A_ASSIGN_URGENT_NEAREST",
    "A_ASSIGN_ANY_NEAREST",
    "A_WAIT",
    "A_REPLAN_TRAFFIC",
    "ALL_ACTIONS",
]
