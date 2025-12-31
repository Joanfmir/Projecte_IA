# main.py
"""Script principal LEGACY para el agente original (no factorizado).

Este script ejecuta la simulación con el agente Q-Learning básico (tabular simple)
o heurísticas simples. Se mantiene como baseline para comparaciones.

Para el agente principal del proyecto (factorizado), usar `main_factored.py`.
"""
from __future__ import annotations
import argparse
import os

from simulation.simulator import Simulator, SimConfig
from simulation.visualizer import Visualizer

from core.dispatch_policy import (
    A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC
)
from core.state_encoding import StateEncoder
from core.q_learning import QLearningAgent


ACTIONS = [A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC]


class TrainedPolicy:
    """Wrapper de política para el agente Q-Learning simple (Legacy).

    Adapta `QLearningAgent` para ser usado en la simulación, utilizando
    el `StateEncoder` original (no factorizado).

    Attributes:
        encoder: Codificador de estado simple.
        agent: Agente Q-Learning cargado desde disco.
    """
    def __init__(self, q_path: str, seed: int = 0):
        self.encoder = StateEncoder()
        self.agent = QLearningAgent.load(q_path)
        self.agent.epsilon = 0.0  # greedy siempre

    # Para loops no visuales
    def choose_action_snapshot(self, snap: dict) -> int:
        """Elige una acción basada en el snapshot (usando el encoder legacy)."""
        s = self.encoder.encode(snap)
        return self.agent.choose_action(s, training=False)


def parse_args():
    """Parsea argumentos de línea de comandos."""
    p = argparse.ArgumentParser(description="Ejecutar agente Legacy (No factorizado)")
    p.add_argument("--visual", action="store_true", help="Abrir simulación con GUI")
    p.add_argument("--policy", choices=["heuristic", "trained"], default="heuristic")
    p.add_argument("--qpath", default="artifacts/qtable.pkl")
    p.add_argument("--width", type=int, default=45)
    p.add_argument("--height", type=int, default=35)
    p.add_argument("--riders", type=int, default=5)
    p.add_argument("--episode_len", type=int, default=900)
    p.add_argument("--spawn", type=float, default=0.1)
    p.add_argument("--max_eta", type=int, default=80)
    p.add_argument("--block_size", type=int, default=6)
    p.add_argument("--street_width", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--interval_ms", type=int, default=240)
    return p.parse_args()


def make_config(a) -> SimConfig:
    """Crea la configuración del simulador."""
    return SimConfig(
        width=a.width,
        height=a.height,
        n_riders=a.riders,
        episode_len=a.episode_len,
        order_spawn_prob=a.spawn,
        max_eta=a.max_eta,
        seed=a.seed,
        block_size=a.block_size,
        street_width=a.street_width,
    )


def run_headless(sim: Simulator, policy_name: str, qpath: str):
    """Ejecuta simulación sin GUI (Legacy)."""
    policy = None
    if policy_name == "trained":
        if not os.path.exists(qpath):
            raise FileNotFoundError(f"No existe {qpath}. Primero ejecuta: python3 train.py")
        policy = TrainedPolicy(qpath)

    done = False
    while not done:
        snap = sim.snapshot()
        if policy_name == "heuristic":
            a = A_ASSIGN_ANY_NEAREST
        else:
            a = policy.choose_action_snapshot(snap)

        _, done = sim.step(a)

    end = sim.snapshot()
    print("FIN")
    print("delivered_total:", end["delivered_total"])
    print("ontime:", end["delivered_ontime"], "late:", end["delivered_late"])
    print("pending_end:", len(end["pending_orders"]))


def run_visual(sim: Simulator, policy_name: str, qpath: str, interval_ms: int):
    """Ejecuta simulación con GUI (Legacy)."""
    if policy_name == "heuristic":
        vis = Visualizer(sim, policy=None, interval_ms=interval_ms)
        vis.run()
        return

    # trained
    if not os.path.exists(qpath):
        raise FileNotFoundError(f"No existe {qpath}. Primero ejecuta: python3 train.py")

    trained = TrainedPolicy(qpath)

    # OJO: para que funcione con tu Visualizer, necesitamos que el Visualizer
    # use choose_action_snapshot() si existe.
    vis = Visualizer(sim, policy=trained, interval_ms=interval_ms)
    vis.run()


def main():
    a = parse_args()
    cfg = make_config(a)
    sim = Simulator(cfg)

    if a.visual:
        run_visual(sim, a.policy, a.qpath, a.interval_ms)
    else:
        run_headless(sim, a.policy, a.qpath)


if __name__ == "__main__":
    main()
