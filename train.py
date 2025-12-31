# train.py
"""Script de entrenamiento LEGACY para el agente original.

Utiliza Q-Learning tabular simple (sin factorización).
Se mantiene por motivos históricos y comparación.
"""
from __future__ import annotations
import os, csv, time
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

from simulation.simulator import Simulator, SimConfig
from core.dispatch_policy import (
    A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC
)
from core.state_encoding import StateEncoder
from core.q_learning import QLearningAgent, QConfig

ACTIONS = [A_ASSIGN_URGENT_NEAREST, A_ASSIGN_ANY_NEAREST, A_WAIT, A_REPLAN_TRAFFIC]


def run_episode(sim: Simulator, agent: QLearningAgent, encoder: StateEncoder, training: bool):
    """Ejecuta un episodio completo con el agente Legacy.

    Args:
        sim: Simulador configurado.
        agent: Agente Q-Learning simple.
        encoder: Codificador de estado simple.
        training: Si es True, actualiza la tabla Q (epsilon-greedy).
                  Si es False, solo evalúa (greedy, epsilon=0).
    """
    total_r = 0.0
    pending_sum = 0
    steps = 0

    fat_sum = 0.0
    resting_sum = 0

    snap = sim.snapshot()
    s = encoder.encode(snap)
    done = False

    while not done:
        snap = sim.snapshot()

        pending_sum += len(snap["pending_orders"])
        steps += 1

        # métricas fatiga
        riders = snap.get("riders", [])
        if riders:
            fat_sum += sum(r.get("fatigue", 0.0) for r in riders) / len(riders)
            resting_sum += sum(1 for r in riders if r.get("resting", False))

        a = agent.choose_action(s, training=training)
        r, done = sim.step(a)
        total_r += r

        snap2 = sim.snapshot()
        s2 = encoder.encode(snap2)

        if training:
            agent.update(s, a, r, s2, done)

        s = s2

    snap_end = sim.snapshot()

    avg_fatigue = fat_sum / max(1, steps)
    pct_resting = (resting_sum / max(1, steps * max(1, len(snap_end.get("riders", []))))) * 100.0

    return {
        "reward": total_r,
        "pending_avg": pending_sum / max(1, steps),
        "delivered_total": snap_end.get("delivered_total", 0),
        "ontime": snap_end.get("delivered_ontime", 0),
        "late": snap_end.get("delivered_late", 0),
        "avg_fatigue": avg_fatigue,
        "pct_resting": pct_resting,
    }


def train(
    n_episodes: int = 100,
    out_dir: str = "artifacts",
    plots_dir: str = "plots",
    eval_every: int = 50,
    eval_episodes: int = 15,
):
    """Entrena el agente Q-Learning simple (Legacy).

    Guarda métricas y gráficas básicas.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # mismo cfg en train y eval 
    cfg = SimConfig(
        width=45, height=35,
        n_riders=6,
        episode_len=900,
        order_spawn_prob=0.40,
        max_eta=80,
        block_size=6,
        street_width=2,
        seed=7,
    )

    encoder = StateEncoder()

    q_path = os.path.join(out_dir, "qtable.pkl")
    metrics_path = os.path.join(out_dir, "metrics.csv")

    # continuar entrenamiento si ya existe Q
    if os.path.exists(q_path):
        agent = QLearningAgent.load(q_path)
    else:
        agent = QLearningAgent(ACTIONS, QConfig(), seed=7)

    t0 = time.time()

    # timestamp para no pisar plots anteriores
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_run_path = os.path.join(out_dir, f"metrics_{run_tag}.csv")

    headers = [
        "episode",
        "reward",
        "pending_avg",
        "delivered_total",
        "ontime",
        "late",
        "avg_fatigue",
        "pct_resting",
        "epsilon",
        "sec_elapsed",
        # eval greedy
        "eval_reward",
        "eval_pending_avg",
        "eval_delivered_total",
        "eval_ontime",
        "eval_late",
    ]

    with open(metrics_run_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for ep in trange(1, n_episodes + 1, desc="Training", unit="ep"):
            sim = Simulator(cfg)

            # ---- TRAIN EPISODE ----
            res = run_episode(sim, agent, encoder, training=True)
            agent.decay_epsilon()

            # ---- EVAL GREEDY (cada eval_every eps) ----
            eval_res = {
                "reward": "",
                "pending_avg": "",
                "delivered_total": "",
                "ontime": "",
                "late": "",
            }
            if eval_every and (ep % eval_every == 0):
                # guardamos epsilon actual
                eps_backup = agent.epsilon
                agent.epsilon = 0.0

                eval_rewards = []
                eval_pending = []
                eval_deliv = []
                eval_ont = []
                eval_late = []

                for _ in range(eval_episodes):
                    sim_eval = Simulator(cfg)
                    r2 = run_episode(sim_eval, agent, encoder, training=False)
                    eval_rewards.append(r2["reward"])
                    eval_pending.append(r2["pending_avg"])
                    eval_deliv.append(r2["delivered_total"])
                    eval_ont.append(r2["ontime"])
                    eval_late.append(r2["late"])

                # medias eval
                eval_res = {
                    "reward": sum(eval_rewards) / len(eval_rewards),
                    "pending_avg": sum(eval_pending) / len(eval_pending),
                    "delivered_total": sum(eval_deliv) / len(eval_deliv),
                    "ontime": sum(eval_ont) / len(eval_ont),
                    "late": sum(eval_late) / len(eval_late),
                }

                # restaurar epsilon
                agent.epsilon = eps_backup

            elapsed = time.time() - t0

            w.writerow([
                ep,
                round(res["reward"], 3),
                round(res["pending_avg"], 3),
                res["delivered_total"],
                res["ontime"],
                res["late"],
                round(res["avg_fatigue"], 4),
                round(res["pct_resting"], 3),
                round(agent.epsilon, 4),
                round(elapsed, 2),

                "" if eval_res["reward"] == "" else round(eval_res["reward"], 3),
                "" if eval_res["pending_avg"] == "" else round(eval_res["pending_avg"], 3),
                "" if eval_res["delivered_total"] == "" else round(eval_res["delivered_total"], 3),
                "" if eval_res["ontime"] == "" else round(eval_res["ontime"], 3),
                "" if eval_res["late"] == "" else round(eval_res["late"], 3),
            ])

            # checkpoint
            if ep % 25 == 0:
                agent.save(q_path)

    agent.save(q_path)

    # además guardo "latest metrics.csv" para que tu plotter por defecto lo lea
    # (copia del run actual)
    try:
        import shutil
        shutil.copy(metrics_run_path, metrics_path)
    except Exception:
        pass

    print("Guardado Q:", q_path)
    print("Metrics run:", metrics_run_path)
    print("Metrics latest:", metrics_path)

    # plots
    plot_metrics(metrics_run_path, plots_dir, run_tag)


def plot_metrics(metrics_csv: str, plots_dir: str, run_tag: str):
    """Genera gráficas básicas de entrenamiento (Legacy)."""
    df = pd.read_csv(metrics_csv)

    def save_plot(y, title, fname):
        plt.figure()
        plt.plot(df["episode"], df[y])
        plt.title(title)
        plt.xlabel("episode")
        plt.ylabel(y)
        plt.tight_layout()
        out = os.path.join(plots_dir, f"{fname}_{run_tag}.png")
        plt.savefig(out, dpi=150)
        plt.close()

    save_plot("delivered_total", "Entregas totales por episodio", "delivered_total")
    save_plot("late", "Entregas tarde", "late")
    save_plot("ontime", "Entregas a tiempo", "ontime")
    save_plot("pending_avg", "Pedidos pendientes (avg)", "pending_avg")
    save_plot("avg_fatigue", "Fatiga media (avg)", "avg_fatigue")
    save_plot("pct_resting", "% de riders descansando", "pct_resting")
    save_plot("reward", "Reward por episodio", "reward")

    # media móvil reward
    window = 50
    if len(df) >= window:
        plt.figure()
        rm = df["reward"].rolling(window=window).mean()
        plt.plot(df["episode"], rm)
        plt.title(f"Reward (media móvil {window})")
        plt.xlabel("episode")
        plt.ylabel(f"reward_avg_{window}")
        plt.tight_layout()
        out = os.path.join(plots_dir, f"reward_avg_{window}_{run_tag}.png")
        plt.savefig(out, dpi=150)
        plt.close()

    # plots eval greedy si existe
    if "eval_reward" in df.columns:
        df_eval = df.dropna(subset=["eval_reward"])
        df_eval = df_eval[df_eval["eval_reward"] != ""]
        if len(df_eval) > 0:
            plt.figure()
            plt.plot(df_eval["episode"], df_eval["eval_reward"])
            plt.title("Eval greedy reward (cada N episodios)")
            plt.xlabel("episode")
            plt.ylabel("eval_reward")
            plt.tight_layout()
            out = os.path.join(plots_dir, f"eval_reward_{run_tag}.png")
            plt.savefig(out, dpi=150)
            plt.close()


if __name__ == "__main__":
    train()
