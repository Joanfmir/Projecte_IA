# train_factored_parallel.py
"""
Entrenamiento paralelo del agente Q-Learning factorizado.
Usa multiprocessing para ejecutar varios episodios simultáneamente.
"""
from __future__ import annotations
import os
import csv
import time
import pickle
from collections import deque
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple, List

from tqdm import tqdm

from simulation.simulator import Simulator, SimConfig
from core.factored_states import FactoredStateEncoder
from core.factored_q_agent import FactoredQAgent, FactoredQConfig
from core.dispatch_policy import A_WAIT


def run_single_episode(args: Tuple) -> Dict:
    """
    Ejecuta un solo episodio y retorna las actualizaciones de Q-table.
    Esta función se ejecuta en un proceso separado.
    """
    (
        ep_seed,
        episode_len,
        base_cfg_dict,
        q1_dict,
        q2_dict,
        q3_dict,
        epsilon,
        cfg_dict,
    ) = args

    # Reconstruir config y simulador
    cfg = SimConfig(**base_cfg_dict)
    cfg.seed = ep_seed
    sim = Simulator(cfg)

    # Crear agente local con Q-tables copiadas
    agent = FactoredQAgent(
        cfg=FactoredQConfig(**cfg_dict),
        encoder=FactoredStateEncoder(episode_len=episode_len),
        seed=ep_seed,
    )
    agent.Q1 = dict(q1_dict)
    agent.Q2 = dict(q2_dict)
    agent.Q3 = dict(q3_dict)
    agent.epsilon = epsilon
    agent.encoder.reset()
    agent.reset_delta()

    total_r = 0.0
    pending_sum = 0
    steps = 0
    q_usage = {"Q1": 0, "Q2": 0, "Q3": 0, "none": 0}

    # Almacenar transiciones para actualizar después
    transitions = []

    snap = sim.snapshot()
    done = False

    while not done:
        pending_sum += len(snap.get("pending_orders", []))
        steps += 1

        action = agent.choose_action(snap, training=True)
        q_usage[agent.last_q_used] += 1
        last_q = agent.last_q_used

        reward, done = sim.step(action)
        reward = reward / 100.0  # Escalar reward para estabilidad
        total_r += reward

        snap_next = sim.snapshot()

        # Guardar transición
        transitions.append((snap, action, reward, snap_next, done, last_q))

        # Actualizar localmente
        agent.update(snap, action, reward, snap_next, done)

        snap = snap_next

    # Métricas
    snap_end = sim.snapshot()

    return {
        "seed": ep_seed,
        "reward": total_r,
        "pending_avg": pending_sum / max(1, steps),
        "delivered_total": snap_end.get("delivered_total", 0),
        "ontime": snap_end.get("delivered_ontime", 0),
        "late": snap_end.get("delivered_late", 0),
        "q_usage": q_usage,
        "max_delta": agent.get_max_delta(),
        # Retornar Q-tables actualizadas
        "Q1": agent.Q1,
        "Q2": agent.Q2,
        "Q3": agent.Q3,
    }


def merge_q_tables(base: Dict, updates: List[Dict], alpha_merge: float = 0.5) -> Dict:
    """
    Combina múltiples Q-tables en una sola.
    Usa promedio ponderado de los valores.
    """
    merged = dict(base)

    for update in updates:
        for key, value in update.items():
            if key in merged:
                # Promedio entre valor existente y nuevo
                merged[key] = merged[key] * (1 - alpha_merge) + value * alpha_merge
            else:
                merged[key] = value

    return merged


def train_parallel(
    n_episodes: int = 500,
    n_workers: int = None,
    batch_size: int = 4,
    out_dir: str = "artifacts",
    base_seed: int = 7,
    episode_len: int = 900,
    delta_threshold: float = 0.01,
    patience: int = 30,
):
    """
    Entrenamiento paralelo con batches de episodios.
    """
    os.makedirs(out_dir, exist_ok=True)

    if n_workers is None:
        n_workers = min(cpu_count(), batch_size)

    print(f"=== Training Paralelo ===")
    print(f"Workers: {n_workers}")
    print(f"Batch size: {batch_size}")
    print(f"Episodios: {n_episodes}")
    print()

    # Config base
    base_cfg = SimConfig(
        width=45,
        height=35,
        n_riders=6,
        episode_len=episode_len,
        order_spawn_prob=0.40,
        max_eta=80,
        block_size=6,
        street_width=2,
        seed=base_seed,
    )

    # Agente principal
    agent_cfg = FactoredQConfig()
    agent = FactoredQAgent(
        cfg=agent_cfg,
        encoder=FactoredStateEncoder(episode_len=episode_len),
        seed=base_seed,
    )

    metrics_path = os.path.join(out_dir, "metrics_factored_parallel.csv")
    q_path = os.path.join(out_dir, "qtable_factored_parallel.pkl")

    t0 = time.time()
    last_rewards = deque(maxlen=50)
    last_deltas = deque(maxlen=50)
    episodes_below_threshold = 0  # Para detección de convergencia

    q_usage_total = {"Q1": 0, "Q2": 0, "Q3": 0, "none": 0}

    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "batch",
                "episodes",
                "reward_avg",
                "reward_avg_50",
                "delta_avg",
                "delta_avg_50",
                "epsilon",
                "Q1_entries",
                "Q2_entries",
                "Q3_entries",
                "sec_elapsed",
            ]
        )

        ep = 0
        batch_num = 0

        with Pool(n_workers) as pool:
            pbar = tqdm(total=n_episodes, desc="Training", unit="ep")

            while ep < n_episodes:
                batch_num += 1
                current_batch_size = min(batch_size, n_episodes - ep)

                # Preparar argumentos para el batch
                args_list = []
                for i in range(current_batch_size):
                    ep_seed = base_seed + ep + i + 1
                    args_list.append(
                        (
                            ep_seed,
                            episode_len,
                            base_cfg.__dict__,
                            dict(agent.Q1),
                            dict(agent.Q2),
                            dict(agent.Q3),
                            agent.epsilon,
                            agent.cfg.__dict__,
                        )
                    )

                # Ejecutar batch en paralelo
                results = pool.map(run_single_episode, args_list)

                # Agregar resultados
                batch_rewards = []
                batch_deltas = []
                q1_updates = []
                q2_updates = []
                q3_updates = []

                for r in results:
                    batch_rewards.append(r["reward"])
                    batch_deltas.append(r["max_delta"])
                    last_rewards.append(r["reward"])
                    last_deltas.append(r["max_delta"])

                    q1_updates.append(r["Q1"])
                    q2_updates.append(r["Q2"])
                    q3_updates.append(r["Q3"])

                    for k in q_usage_total:
                        q_usage_total[k] += r["q_usage"][k]

                # Merge Q-tables
                agent.Q1 = merge_q_tables(agent.Q1, q1_updates, alpha_merge=0.3)
                agent.Q2 = merge_q_tables(agent.Q2, q2_updates, alpha_merge=0.3)
                agent.Q3 = merge_q_tables(agent.Q3, q3_updates, alpha_merge=0.3)

                # Decay epsilon (una vez por batch)
                for _ in range(current_batch_size):
                    agent.decay_epsilon()

                ep += current_batch_size
                pbar.update(current_batch_size)

                # Métricas
                reward_avg = sum(batch_rewards) / len(batch_rewards)
                delta_avg = sum(batch_deltas) / len(batch_deltas)
                reward_avg_50 = sum(last_rewards) / len(last_rewards)
                delta_avg_50 = sum(last_deltas) / len(last_deltas)
                elapsed = time.time() - t0

                w.writerow(
                    [
                        batch_num,
                        ep,
                        round(reward_avg, 3),
                        round(reward_avg_50, 3),
                        round(delta_avg, 4),
                        round(delta_avg_50, 4),
                        round(agent.epsilon, 4),
                        len(agent.Q1),
                        len(agent.Q2),
                        len(agent.Q3),
                        round(elapsed, 2),
                    ]
                )
                f.flush()

                # Checkpoint cada 25 episodios
                if ep % 25 == 0:
                    agent.save(q_path)
                    pbar.set_postfix(
                        {
                            "delta": f"{delta_avg_50:.1f}",
                            "eps": f"{agent.epsilon:.2f}",
                        }
                    )

                # Detección de convergencia
                if delta_avg_50 < delta_threshold and len(last_deltas) >= 50:
                    episodes_below_threshold += current_batch_size
                    if episodes_below_threshold >= patience:
                        print(
                            f"\n¡CONVERGENCIA alcanzada en ep {ep}! delta_avg_50={delta_avg_50:.4f}"
                        )
                        break
                else:
                    episodes_below_threshold = 0

            pbar.close()

    agent.save(q_path)
    print(f"\nGuardado: {q_path}")
    print(f"Guardado: {metrics_path}")
    print(f"\nQ-usage total: {q_usage_total}")
    print(f"Stats finales: {agent.stats()}")
    print(f"Tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    train_parallel(
        n_episodes=args.episodes,
        n_workers=args.workers,
        batch_size=args.batch,
        base_seed=args.seed,
    )
