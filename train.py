"""
Wrapper legacy que delega al flujo factorizado.
Conserva los nombres/constantes para compatibilidad.
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
from train_factored import train as train_factored

ACTIONS = [A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC]


def train(*args, **kwargs):
    """Wrapper deprecado que llama a train_factored.train."""
    warnings.warn(
        "train.py es legacy; usa train_factored.train o ejecuta train_factored.py",
        DeprecationWarning,
        stacklevel=2,
    )
    return train_factored(*args, **kwargs)


def main():
    warnings.warn(
        "train.py es legacy; redirigiendo a train_factored.py",
        DeprecationWarning,
        stacklevel=2,
    )
    sys.argv[0] = "train_factored.py"
    runpy.run_module("train_factored", run_name="__main__")


if __name__ == "__main__":
    main()
