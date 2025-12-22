# train.py (solo necesitas esto; si prefieres te lo adapto a tu archivo exacto)
from __future__ import annotations
import os, csv, time

from tqdm import trange  # <-- NUEVO

from simulation.simulator import Simulator, SimConfig
from core.dispatch_policy import A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC
from core.state_encoding import StateEncoder
from core.q_learning import QLearningAgent, QConfig

ACTIONS = [A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC]


def train(n_episodes: int = 600, out_dir: str = "artifacts"):
    os.makedirs(out_dir, exist_ok=True)

    cfg = SimConfig(
        width=45, height=35,
        n_riders=6,
        episode_len=900,
        order_spawn_prob=0.35,
        max_eta=80,
        block_size=6,
        street_width=2,
        seed=7,
    )

    encoder = StateEncoder()
    agent = QLearningAgent(ACTIONS, QConfig(), seed=7)

    metrics_path = os.path.join(out_dir, "metrics.csv")
    q_path = os.path.join(out_dir, "qtable.pkl")

    t0 = time.time()

    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "pending_avg", "delivered_total", "ontime", "late", "epsilon", "sec_elapsed"])

        # ✅ barra de progreso con ETA
        for ep in trange(1, n_episodes + 1, desc="Training", unit="ep"):
            sim = Simulator(cfg)

            total_r = 0.0
            pending_sum = 0
            steps = 0

            s = encoder.encode(sim.snapshot())
            done = False

            while not done:
                snap = sim.snapshot()
                pending_sum += len(snap["pending_orders"])
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

            w.writerow([
                ep,
                round(total_r, 3),
                round(pending_avg, 3),
                snap_end.get("delivered_total", 0),
                snap_end.get("delivered_ontime", 0),
                snap_end.get("delivered_late", 0),
                round(agent.epsilon, 4),
                round(elapsed, 2),
            ])

            # cada 25 episodios guardamos checkpoint
            if ep % 25 == 0:
                agent.save(q_path)

    agent.save(q_path)
    print("Guardado:", q_path)
    print("Guardado:", metrics_path)


if __name__ == "__main__":
    train()
