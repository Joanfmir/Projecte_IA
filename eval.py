# eval.py
"""Script de evaluación LEGACY para el agente original.

Compara el agente entrenado sin factorización contra una heurística simple.
Muestra métricas en consola.
"""
from __future__ import annotations
import statistics

from simulation.simulator import Simulator, SimConfig
from core.dispatch_policy import (
    A_ASSIGN_ANY_NEAREST, A_ASSIGN_URGENT_NEAREST, A_WAIT, A_REPLAN_TRAFFIC
)
from core.state_encoding import StateEncoder
from core.q_learning import QLearningAgent

ACTIONS = [A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC]


def run_episode(sim: Simulator, policy_fn):
    """Ejecuta un episodio de evaluación (sin entrenar)."""
    total_r = 0.0
    pending_sum = 0
    steps = 0

    done = False
    while not done:
        snap = sim.snapshot()
        pending_sum += len(snap.get("pending_orders", []))
        steps += 1

        a = policy_fn(sim, snap)
        r, done = sim.step(a)
        total_r += r

    snap_end = sim.snapshot()
    return {
        "reward": total_r,
        "pending_avg": pending_sum / max(1, steps),
        "delivered_total": snap_end.get("delivered_total", 0),
        "ontime": snap_end.get("delivered_ontime", 0),
        "late": snap_end.get("delivered_late", 0),
    }


def eval_all(n_episodes: int = 40, q_path: str = "artifacts/qtable.pkl", base_seed: int = 999):
    """Ejecuta una evaluación comparativa (Heurística vs Q-Learning)."""
    base_cfg = SimConfig(
        width=45, height=35,
        n_riders=6,
        episode_len=900,
        order_spawn_prob=0.35,
        max_eta=80,
        block_size=6,
        street_width=2,
        seed=base_seed,
    )

    encoder = StateEncoder()
    agent = QLearningAgent.load(q_path)
    agent.epsilon = 0.0  # greedy en evaluación

    def heuristic_policy(sim, snap):
        return A_ASSIGN_ANY_NEAREST

    def q_policy(sim, snap):
        s = encoder.encode(snap)
        return agent.choose_action(s, training=False)

    results_h = []
    results_q = []

    for i in range(n_episodes):
        cfg = SimConfig(**base_cfg.__dict__)
        cfg.seed = base_seed + i  # ✅ cada episodio eval con mapa/azar distinto

        results_h.append(run_episode(Simulator(cfg), heuristic_policy))
        results_q.append(run_episode(Simulator(cfg), q_policy))

    def summarize(name, res):
        print("\n===", name, "===")
        for k in ["reward", "pending_avg", "delivered_total", "ontime", "late"]:
            vals = [x[k] for x in res]
            print(f"{k:14s}: mean={statistics.mean(vals):.3f}  std={statistics.pstdev(vals):.3f}")

    summarize("HEURISTIC (nearest)", results_h)
    summarize("Q-LEARNING", results_q)


if __name__ == "__main__":
    eval_all()
