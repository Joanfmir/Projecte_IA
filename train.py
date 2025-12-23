# train.py
from __future__ import annotations
import os, csv, time
from collections import deque

from tqdm import trange

from simulation.simulator import Simulator, SimConfig
from core.dispatch_policy import (
    A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC
)
from core.state_encoding import StateEncoder
from core.q_learning import QLearningAgent, QConfig

ACTIONS = [A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC]


def train(n_episodes: int = 600, out_dir: str = "artifacts", base_seed: int = 7):
    os.makedirs(out_dir, exist_ok=True)

    # ⚙️ Config base (NO fijamos seed aquí para todos los episodios)
    base_cfg = SimConfig(
        width=45, height=35,
        n_riders=6,
        episode_len=900,
        order_spawn_prob=0.40,
        max_eta=80,
        block_size=6,
        street_width=2,
        seed=base_seed,  # se sobrescribe por episodio
    )

    encoder = StateEncoder()
    agent = QLearningAgent(ACTIONS, QConfig(), seed=base_seed)

    metrics_path = os.path.join(out_dir, "metrics.csv")
    q_path = os.path.join(out_dir, "qtable.pkl")

    t0 = time.time()

    # para suavizar la curva y ver si mejora
    last_rewards = deque(maxlen=50)

    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "episode",
            "reward",
            "reward_avg_50",
            "pending_avg",
            "delivered_total",
            "ontime",
            "late",
            "epsilon",
            "sec_elapsed",
            "seed",
        ])

        for ep in trange(1, n_episodes + 1, desc="Training", unit="ep"):
            # ✅ SEED distinta por episodio (clave)
            cfg = SimConfig(**base_cfg.__dict__)
            cfg.seed = base_seed + ep  # cambia mapa/aleatoriedad

            sim = Simulator(cfg)

            total_r = 0.0
            pending_sum = 0
            steps = 0

            s = encoder.encode(sim.snapshot())
            done = False

            while not done:
                snap = sim.snapshot()
                pending_sum += len(snap.get("pending_orders", []))
                steps += 1

                a = agent.choose_action(s, training=True)
                r, done = sim.step(a)
                total_r += r

                s2 = encoder.encode(sim.snapshot())
                agent.update(s, a, r, s2, done)
                s = s2

            agent.decay_epsilon()

            snap_end = sim.snapshot()
            pending_avg = pending_sum / max(1, steps)
            elapsed = time.time() - t0

            last_rewards.append(total_r)
            reward_avg_50 = sum(last_rewards) / len(last_rewards)

            w.writerow([
                ep,
                round(total_r, 3),
                round(reward_avg_50, 3),
                round(pending_avg, 3),
                snap_end.get("delivered_total", 0),
                snap_end.get("delivered_ontime", 0),
                snap_end.get("delivered_late", 0),
                round(agent.epsilon, 4),
                round(elapsed, 2),
                cfg.seed,
            ])

            # ✅ flush para que se vaya escribiendo aunque pares con Ctrl+C
            f.flush()

            # checkpoint cada 25 episodios
            if ep % 25 == 0:
                agent.save(q_path)

    agent.save(q_path)
    print("Guardado:", q_path)
    print("Guardado:", metrics_path)


if __name__ == "__main__":
    train()
