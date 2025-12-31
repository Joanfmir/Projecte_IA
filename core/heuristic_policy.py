# core/heuristic_policy.py
"""Política heurística operando sobre snapshots del simulador.

Provee la misma interfaz que `FactoredQAgent` para compatibilidad con el
visualizador y los scripts de ejecución.
"""
from __future__ import annotations

from core.shared_params import A_ASSIGN_ANY_NEAREST


class HeuristicPolicy:
    """Política baseline simple que siempre asigna al rider más cercano disponible."""

    def choose_action_snapshot(self, _snap: dict) -> int:
        """Devuelve la acción heurística: asignar al más cercano."""
        return A_ASSIGN_ANY_NEAREST


__all__ = ["HeuristicPolicy"]
