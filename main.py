"""
Legacy wrapper to keep compatibility and redirect to main_factored.py.
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
from main_factored import TrainedFactoredPolicy as TrainedPolicy

ACTIONS = [A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC]


def main():
    warnings.warn(
        "main.py is legacy; redirecting to main_factored.py",
        DeprecationWarning,
        stacklevel=2,
    )
    sys.argv[0] = "main_factored.py"
    runpy.run_module("main_factored", run_name="__main__")


if __name__ == "__main__":
    main()
