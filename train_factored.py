# train_factored.py
"""
Entrenamiento del agente Q-Learning factorizado.
Con detección de convergencia por delta Q.
"""
from __future__ import annotations
import os
import csv
import time
from collections import deque

from tqdm import trange

from simulation.simulator import Simulator, SimConfig
from core.dispatch_policy import A_WAIT
from core.factored_states import FactoredStateEncoder
from core.factored_q_agent import FactoredQAgent, FactoredQConfig


def train(
    n_episodes: int = 500,
    out_dir: str = "artifacts",
    base_seed: int = 7,
    episode_len: int = 900,
    flush_every: int = 50,
    checkpoint_every: int = 50,
    fast: bool = False,
    # Convergencia
    delta_threshold: float = 0.01,  # Umbral de delta Q para convergencia
    patience: int = 30,  # Episodios consecutivos bajo umbral para parar
):
    os.makedirs(out_dir, exist_ok=True)

    if fast:
        n_episodes = min(n_episodes, 2)
        episode_len = min(episode_len, 200)
    flush_every = max(1, flush_every)
    checkpoint_every = max(1, checkpoint_every)

    # Config del simulador
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
        enable_internal_spawn=True,
        enable_internal_traffic=True,
    )

    # Agente factorizado
    agent = FactoredQAgent(
        cfg=FactoredQConfig(),
        encoder=FactoredStateEncoder(episode_len=episode_len),
        seed=base_seed,
    )

    metrics_path = os.path.join(out_dir, "metrics_factored.csv")
    q_path = os.path.join(out_dir, "qtable_factored.pkl")

    t0 = time.time()
    last_rewards = deque(maxlen=50)
    last_deltas = deque(maxlen=50)  # Para suavizar delta Q

    # Contadores de uso de Q-tables
    q_usage = {"Q1": 0, "Q3": 0, "none": 0}

    # Convergencia
    converged = False
    episodes_below_threshold = 0

    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "episode",
                "reward",
                "reward_avg_50",
                "pending_avg",
                "delivered_total",
                "ontime",
                "late",
                "epsilon",
                "max_delta_q",
                "delta_avg_50",
                "sec_elapsed",
                "seed",
                "Q1_used",
                "Q3_used",
                "Q1_entries",
                "Q3_entries",
                "ticks_total",
                "wait_count",
                "action_count",
                "wait_ratio",
                "avg_rider_load",
                "batching_efficiency",
                "activated_riders",
            ]
        )

        for ep in trange(1, n_episodes + 1, desc="Training", unit="ep"):
            # Seed distinta por episodio
            cfg = SimConfig(**base_cfg.__dict__)
            cfg.seed = base_seed + ep

            sim = Simulator(cfg)
            should_break = False
            try:
                agent.encoder.reset()  # Reset estado interno del encoder
                agent.reset_delta()  # Reset max_delta_q del episodio

                total_r = 0.0
                pending_sum = 0
                steps = 0
                ep_q_usage = {"Q1": 0, "Q3": 0, "none": 0}
                wait_count = 0
                action_count = 0
                load_positive_sum = 0
                load_positive_count = 0
                ticks_with_batching = 0
                activated_riders = set()

                snapshot_fn = sim.snapshot
                step_fn = sim.step
                choose_action = agent.choose_action
                update_agent = agent.update
                commit_encoder = agent.commit_encoder

                snap = snapshot_fn()
                done = False

                while not done:
                    pending_orders = snap.get("pending_orders", [])
                    riders = snap.get("riders", [])

                    # Métricas de batching
                    positive_loads = 0
                    positive_sum = 0
                    batching_tick = False
                    for r in riders:
                        assigned = set(r.get("assigned", []))
                        load = len(assigned)
                        carrying = r.get("carrying")
                        if carrying is not None and carrying not in assigned:
                            load += 1
                        if load > 0:
                            positive_loads += 1
                            positive_sum += load
                            activated_riders.add(r.get("id"))
                            if load >= 2:
                                batching_tick = True
                    if positive_loads:
                        load_positive_sum += positive_sum
                        load_positive_count += positive_loads
                    if batching_tick:
                        ticks_with_batching += 1

                    pending_sum += len(pending_orders)
                    steps += 1

                    # Elegir acción (no modifica prev_traffic_pressure)
                    action = choose_action(snap, training=True)
                    action_count += 1
                    if action == A_WAIT:
                        wait_count += 1
                    ep_q_usage[agent.last_q_used] += 1

                    # Ejecutar paso
                    reward, done = step_fn(action)
                    total_r += reward

                    # Siguiente snapshot
                    snap_next = snapshot_fn()

                    # Actualizar Q-table (esto actualiza max_delta_q internamente)
                    update_agent(snap, action, reward, snap_next, done)

                    # Commit: actualizar prev_traffic_pressure con lo que OBSERVAMOS (snap)
                    # Así en el siguiente tick, delta_traffic = |nuevo - snap| detecta cambios
                    commit_encoder(snap)

                    snap = snap_next

                # Decay epsilon al final del episodio
                agent.decay_epsilon()

                # Acumular uso de Q-tables
                for k in q_usage:
                    q_usage[k] += ep_q_usage[k]

                # Métricas del episodio
                snap_end = sim.snapshot()
                pending_avg = pending_sum / max(1, steps)
                elapsed = time.time() - t0
                max_delta = agent.get_max_delta()

                last_rewards.append(total_r)
                last_deltas.append(max_delta)
                reward_avg_50 = sum(last_rewards) / len(last_rewards)
                delta_avg_50 = sum(last_deltas) / len(last_deltas)
                wait_ratio = wait_count / action_count if action_count else 0.0
                avg_rider_load = (
                    load_positive_sum / load_positive_count
                    if load_positive_count
                    else 0.0
                )
                batching_efficiency = (
                    ticks_with_batching / steps if steps else 0.0
                )
                activated_riders_count = len(activated_riders)

                stats = agent.stats()

                w.writerow(
                    [
                        ep,
                        round(total_r, 3),
                        round(reward_avg_50, 3),
                        round(pending_avg, 3),
                        snap_end.get("delivered_total", 0),
                        snap_end.get("delivered_ontime", 0),
                        snap_end.get("delivered_late", 0),
                        round(agent.epsilon, 4),
                        round(max_delta, 6),
                        round(delta_avg_50, 6),
                        round(elapsed, 2),
                        cfg.seed,
                        ep_q_usage["Q1"],
                        ep_q_usage["Q3"],
                        stats["Q1_entries"],
                        stats["Q3_entries"],
                        steps,
                        wait_count,
                        action_count,
                        round(wait_ratio, 4),
                        round(avg_rider_load, 4),
                        round(batching_efficiency, 4),
                        activated_riders_count,
                    ]
                )

                if (ep % flush_every) == 0:
                    f.flush()

                # Checkpoint periódico
                if (ep % checkpoint_every) == 0:
                    agent.save(q_path)
                    print(
                        f"\n[Ep {ep}] max_delta={max_delta:.4f}, delta_avg={delta_avg_50:.4f}, eps={agent.epsilon:.3f}"
                    )

                # === Detección de convergencia ===
                if ep >= 50:  # Esperar al menos 50 episodios
                    if delta_avg_50 < delta_threshold:
                        episodes_below_threshold += 1
                        if episodes_below_threshold >= patience:
                            print(f"\n[OK] CONVERGENCIA alcanzada en episodio {ep}")
                            print(
                                f"   delta_avg_50 = {delta_avg_50:.6f} < {delta_threshold}"
                            )
                            converged = True
                            should_break = True
                    else:
                        episodes_below_threshold = 0
            finally:
                # Simulator no expone método explícito de cierre; liberamos la referencia
                del sim
            if should_break:
                break

    # Guardar al final
    agent.save(q_path)
    print(f"\nGuardado: {q_path}")
    print(f"Guardado: {metrics_path}")
    print(f"\nUso total de Q-tables: {q_usage}")
    print(f"Stats finales: {agent.stats()}")

    if not converged:
        print(
            f"\n[!] No convergió en {n_episodes} episodios. delta_avg_50 final = {delta_avg_50:.6f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Entrenamiento secuencial Q-Learning factorizado"
    )
    parser.add_argument("--episodes", type=int, default=500, help="Número de episodios")
    parser.add_argument("--seed", type=int, default=7, help="Seed base")
    parser.add_argument(
        "--episode_len", type=int, default=900, help="Longitud de episodio"
    )
    parser.add_argument(
        "--delta", type=float, default=0.01, help="Umbral delta para convergencia"
    )
    parser.add_argument(
        "--patience", type=int, default=30, help="Episodios bajo umbral para parar"
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=50,
        help="Frecuencia de flush a disco (episodios)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Frecuencia de guardado de Q-table (episodios)",
    )
    parser.add_argument(
        "--fast",
        "--debug",
        action="store_true",
        help="Modo rápido: 2 episodios de 200 ticks para smoke test",
    )
    args = parser.parse_args()

    train(
        n_episodes=args.episodes,
        base_seed=args.seed,
        episode_len=args.episode_len,
        flush_every=args.flush_every,
        checkpoint_every=args.checkpoint_every,
        fast=args.fast,
        delta_threshold=args.delta,
        patience=args.patience,
    )
