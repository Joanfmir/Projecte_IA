# Legacy removal inventory and actions (factored-only)

## Inventory (before removal)
- `simulation/simulator.py` :: `compute_state` (legacy state binning) :: called only by `simulation/visualizer.py` fallback.
- `simulation/visualizer.py` :: fallback `policy.choose_action` + `sim.compute_state` (legacy policy path).
- `core/dispatch_policy.py` :: legacy tabular policy/QL config/constants (unused by factored flow).
- `core/q_learning.py` :: legacy Q-learning agent (unused by factored flow).
- `core/state_encoding.py` :: legacy encoder for tabular agent (unused by factored flow).
- Entry wrappers `train.py`, `eval.py`, `main.py` :: legacy compatibility paths.

## Removal/mitigation (after changes)
- Removed `compute_state` from `Simulator` and the visualizer fallback; visualizer now requires `choose_action_snapshot` or raises a clear `TypeError`.
- Deleted legacy modules: `core/dispatch_policy.py`, `core/q_learning.py`, `core/state_encoding.py`.
- Simplified `core/shared_params.py` to action constants only (single source of truth for action ids).
- Added `core/heuristic_policy.HeuristicPolicy` exposing `choose_action_snapshot`.
- Legacy entrypoints `train.py`, `eval.py`, `main.py` now exit immediately with guidance to factored scripts.

## Files removed
- `core/dispatch_policy.py`
- `core/q_learning.py`
- `core/state_encoding.py`

## Factored entrypoints
- Training: `train_factored.py`
- Evaluation: `eval_factored.py`
- Runner/UI: `main_factored.py`

## Supported policies (snapshot-based)
- FactoredQAgent (trained)
- HeuristicPolicy (pure heuristic)
