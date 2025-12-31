"""
Wrapper legacy que delega a la evaluación factorizada.
Mantiene constantes y firma compatible para código existente.
"""
from __future__ import annotations

import runpy
import sys
import warnings

from core.shared_params import (
    A_ASSIGN_ANY_NEAREST,
    A_ASSIGN_URGENT_NEAREST,
    A_REPLAN_TRAFFIC,
    A_WAIT,
)
from eval_factored import EvalConfig, evaluate as evaluate_factored

ACTIONS = [A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC]


def eval_all(
    n_episodes: int = 20,
    q_path: str = "artifacts/qtable_factored.pkl",
    base_seed: int = 7,
):
    """
    Wrapper legacy que redirige a evaluate() del flujo factorizado.
    """
    warnings.warn(
        "eval.py es legacy; usa eval_factored.evaluate/main",
        DeprecationWarning,
        stacklevel=2,
    )
    cfg = EvalConfig(n_episodes=n_episodes, q_path=q_path, base_seed=base_seed)
    return evaluate_factored(cfg)


def main():
    warnings.warn(
        "eval.py es legacy; redirigiendo a eval_factored.py",
        DeprecationWarning,
        stacklevel=2,
    )
    sys.argv[0] = "eval_factored.py"
    runpy.run_module("eval_factored", run_name="__main__")


if __name__ == "__main__":
    main()
