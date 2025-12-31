"""
Legacy wrapper delegating to the factored training flow.
Keeps action constants for compatibility.
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
    """Deprecated wrapper that forwards to train_factored.train."""
    warnings.warn(
        "train.py is legacy; use the factored flow (train_factored.py)",
        DeprecationWarning,
        stacklevel=2,
    )
    return train_factored(*args, **kwargs)


def main():
    warnings.warn(
        "train.py is legacy; redirecting to train_factored.py",
        DeprecationWarning,
        stacklevel=2,
    )
    sys.argv[0] = "train_factored.py"
    runpy.run_module("train_factored", run_name="__main__")


if __name__ == "__main__":
    main()
