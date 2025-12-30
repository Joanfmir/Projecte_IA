# train_factored.py
"""
Entrenamiento del agente Q-Learning factorizado.
Con detección de convergencia por delta Q.
"""
from __future__ import annotations
import copy
import multiprocessing as mp
import os
import csv
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from tqdm import trange, tqdm

from simulation.simulator import Simulator, SimConfig
from core.dispatch_policy import A_WAIT
from core.factored_states import FactoredStateEncoder
from core.factored_q_agent import FactoredQAgent, FactoredQConfig

FAST_EPISODES = 2
FAST_MAX_TICKS = 200
WORKER_SEED_STRIDE = 9973  # Primo para espaciar seeds entre episodios/worker
EVAL_SEED_OFFSET = 100_000  # Evita solaparse con semillas de entrenamiento


@dataclass
class ParallelTrainConfig:
    """Configuración del entrenamiento paralelo."""

    n_episodes: int
    out_dir: str
    base_seed: int
    episode_len: int
    max_steps_per_episode: int
    n_workers: int
    epsilon_start: float | None
    epsilon_end: float
    epsilon_decay_steps: int
    eval_every: int
    eval_episodes: int
    log_every: int
    save_every: int
    sync_every: int
    fast: bool = False
    init_q_path: str | None = None
    q_path: str = "artifacts/qtable_factored.pkl"
    metrics_path: str = "artifacts/metrics_factored_parallel.csv"
    alpha: float = 0.1
    gamma: float = 0.95


@dataclass
class EpisodeResult:
    """Resultado de un episodio generado en un worker."""

    episode: int
    seed: int
    reward: float
    pending_sum: int
    steps: int
    wait_count: int
    action_count: int
    load_positive_sum: int
    load_positive_count: int
    ticks_with_batching: int
    activated_riders: int
    delivered_total: int
    delivered_ontime: int
    delivered_late: int
    transitions: List[
        Tuple[Dict[str, Any], int, float, Dict[str, Any], bool, str]
    ]
    q_usage: Dict[str, int]
    snap_end: Dict[str, Any]


def _build_base_sim_config(episode_len: int, seed: int) -> SimConfig:
    """Config base compartida con train secuencial."""
    return SimConfig(
        width=45,
        height=35,
        n_riders=6,
        episode_len=episode_len,
        order_spawn_prob=0.40,
        max_eta=80,
        block_size=6,
        street_width=2,
        seed=seed,
        enable_internal_spawn=True,
        enable_internal_traffic=True,
    )


def _epsilon_scheduler(
    epsilon_start: float, epsilon_end: float, decay_steps: int
) -> Callable[[int], float]:
    """Crea una schedule monótona y acotada."""
    epsilon_start = float(epsilon_start)
    epsilon_end = float(epsilon_end)
    if decay_steps <= 0:
        return lambda _: epsilon_start

    span = max(1, decay_steps)

    def schedule(step_index: int) -> float:
        idx = min(max(step_index, 0), span)
        eps = epsilon_start - (epsilon_start - epsilon_end) * (idx / span)
        return eps

    return schedule


def _snapshot_agent(agent: FactoredQAgent) -> Dict[str, Any]:
    """Snapshot inmutable para workers."""
    return {
        "Q1": copy.deepcopy(agent.Q1),
        "Q3": copy.deepcopy(agent.Q3),
        "cfg": agent.cfg,
        "actions_q1": list(agent.actions_q1),
        "actions_q3": list(agent.actions_q3),
    }


def _make_worker_agent(
    snapshot: Dict[str, Any], epsilon: float, episode_len: int, seed: int
) -> FactoredQAgent:
    """Construye un agente para rollout (solo lee)."""
    agent = FactoredQAgent(
        cfg=snapshot["cfg"],
        encoder=FactoredStateEncoder(episode_len=episode_len),
        seed=seed,
    )
    agent.Q1 = copy.deepcopy(snapshot["Q1"])
    agent.Q3 = copy.deepcopy(snapshot["Q3"])
    agent.actions_q1 = snapshot.get("actions_q1", agent.actions_q1)
    agent.actions_q3 = snapshot.get("actions_q3", agent.actions_q3)
    agent.epsilon = epsilon
    agent.encoder.reset()
    agent.reset_delta()
    return agent


def _run_episode_worker(
    episode: int,
    base_cfg_dict: Dict[str, Any],
    snapshot: Dict[str, Any],
    epsilon: float,
    max_steps: int,
    base_seed: int,
) -> EpisodeResult:
    """Ejecuta un episodio completo y retorna transiciones."""
    cfg = SimConfig(**base_cfg_dict)
    cfg.seed = base_seed + episode
    cfg.episode_len = max_steps

    agent_seed = base_seed + episode * WORKER_SEED_STRIDE
    agent = _make_worker_agent(snapshot, epsilon, max_steps, agent_seed)

    sim = Simulator(cfg)
    total_r = 0.0
    pending_sum = 0
    steps = 0
    wait_count = 0
    action_count = 0
    load_positive_sum = 0
    load_positive_count = 0
    ticks_with_batching = 0
    activated_riders = set()
    q_usage = {"Q1": 0, "Q3": 0, "none": 0}
    transitions: List[
        Tuple[Dict[str, Any], int, float, Dict[str, Any], bool, str]
    ] = []

    snap = sim.snapshot()
    done = False

    while not done:
        riders = snap.get("riders", [])

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

        pending_sum += len(snap.get("pending_orders", []))
        steps += 1

        action = agent.choose_action(snap, training=True)
        q_usage[agent.last_q_used] += 1
        if action == A_WAIT:
            wait_count += 1
        action_count += 1

        reward, done = sim.step(action)
        total_r += reward

        snap_next = sim.snapshot()
        transitions.append((snap, action, reward, snap_next, done, agent.last_q_used))

        agent.commit_encoder(snap)
        snap = snap_next

        if steps >= max_steps:
            break

    snap_end = sim.snapshot()

    return EpisodeResult(
        episode=episode,
        seed=cfg.seed,
        reward=total_r,
        pending_sum=pending_sum,
        steps=steps,
        wait_count=wait_count,
        action_count=action_count,
        load_positive_sum=load_positive_sum,
        load_positive_count=load_positive_count,
        ticks_with_batching=ticks_with_batching,
        activated_riders=len(activated_riders),
        delivered_total=snap_end.get("delivered_total", 0),
        delivered_ontime=snap_end.get("delivered_ontime", 0),
        delivered_late=snap_end.get("delivered_late", 0),
        transitions=transitions,
        q_usage=q_usage,
        snap_end=snap_end,
    )


def _apply_episode_to_agent(
    agent: FactoredQAgent, result: EpisodeResult, epsilon: float
) -> Dict[str, Any]:
    """Aplica transiciones en el learner en orden estable."""
    agent.encoder.reset()
    agent.reset_delta()
    agent.epsilon = epsilon

    for snap, action, reward, snap_next, done, q_used in result.transitions:
        agent.last_q_used = q_used
        agent.update(snap, action, reward, snap_next, done)
        agent.commit_encoder(snap)

    stats = agent.stats()
    max_delta = agent.get_max_delta()
    agent.encoder.reset()
    return {
        "max_delta": max_delta,
        "stats": stats,
    }


def _evaluate_greedy(
    agent: FactoredQAgent,
    base_cfg: SimConfig,
    n_episodes: int,
    base_seed: int,
) -> Dict[str, float]:
    """Evalúa con epsilon=0 y restaura después."""
    prev_eps = agent.epsilon
    agent.epsilon = 0.0

    rewards = []
    wait_ratios = []
    batching_eff = []
    avg_loads = []

    for i in range(n_episodes):
        cfg = SimConfig(**base_cfg.__dict__)
        cfg.seed = base_seed + EVAL_SEED_OFFSET + i
        agent.rng = random.Random(cfg.seed)  # garantiza desempates deterministas en eval

        sim = Simulator(cfg)
        total_r = 0.0
        wait_count = 0
        action_count = 0
        load_positive_sum = 0
        load_positive_count = 0
        ticks_with_batching = 0
        steps = 0

        snap = sim.snapshot()
        done = False
        agent.encoder.reset()
        while not done:
            riders = snap.get("riders", [])
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
                    if load >= 2:
                        batching_tick = True
            if positive_loads:
                load_positive_sum += positive_sum
                load_positive_count += positive_loads
            if batching_tick:
                ticks_with_batching += 1

            action = agent.choose_action(snap, training=False)
            if action == A_WAIT:
                wait_count += 1
            action_count += 1
            steps += 1

            reward, done = sim.step(action)
            total_r += reward
            snap_next = sim.snapshot()
            agent.commit_encoder(snap)
            snap = snap_next

        snap_end = sim.snapshot()
        rewards.append(total_r)
        wait_ratios.append(wait_count / action_count if action_count else 0.0)
        avg_loads.append(
            load_positive_sum / load_positive_count if load_positive_count else 0.0
        )
        ticks_total = snap_end.get("t", steps)
        batching_eff.append(ticks_with_batching / max(1, ticks_total))

    agent.epsilon = prev_eps
    return {
        "reward_avg": sum(rewards) / len(rewards) if rewards else 0.0,
        "wait_ratio_avg": sum(wait_ratios) / len(wait_ratios) if wait_ratios else 0.0,
        "batching_eff_avg": sum(batching_eff) / len(batching_eff)
        if batching_eff
        else 0.0,
        "avg_rider_load": sum(avg_loads) / len(avg_loads) if avg_loads else 0.0,
    }


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
    # Hiperparámetros Q-Learning
    alpha: float = 0.1,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_decay: float = 0.995,
    eps_min: float = 0.05,
):
    os.makedirs(out_dir, exist_ok=True)

    if fast:
        n_episodes = min(n_episodes, FAST_EPISODES)
        episode_len = min(episode_len, FAST_MAX_TICKS)
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

    # Agente factorizado con hiperparámetros personalizables
    agent = FactoredQAgent(
        cfg=FactoredQConfig(
            alpha=alpha,
            gamma=gamma,
            eps_start=eps_start,
            eps_decay=eps_decay,
            eps_min=eps_min,
        ),
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
                    if action == A_WAIT:
                        wait_count += 1
                    action_count += 1
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
                ticks_total = snap_end.get("t", steps)

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
                    ticks_with_batching / ticks_total if ticks_total else 0.0
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
                # Simulator no ofrece un método explícito de cierre; liberamos la referencia
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


def train_parallel(cfg: ParallelTrainConfig) -> None:
    """Entrenamiento en modo actor-learner con multiprocessing."""
    os.makedirs(cfg.out_dir, exist_ok=True)

    n_episodes = cfg.n_episodes
    max_steps = cfg.max_steps_per_episode
    if cfg.fast:
        n_episodes = min(n_episodes, FAST_EPISODES)
        max_steps = min(max_steps, FAST_MAX_TICKS)

    try:
        current_method = mp.get_start_method(allow_none=True)
        if current_method != "spawn":
            # spawn evita compartir estado/PRNG del simulador entre procesos
            mp.set_start_method("spawn")
    except RuntimeError:
        # Ya estaba configurado
        pass

    base_cfg = _build_base_sim_config(max_steps, cfg.base_seed)
    metrics_path = cfg.metrics_path
    q_path = cfg.q_path
    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    q_dir = os.path.dirname(q_path)
    if q_dir:
        os.makedirs(q_dir, exist_ok=True)

    # Cargar agente (compatibilidad con checkpoints existentes)
    if cfg.init_q_path and os.path.exists(cfg.init_q_path):
        agent = FactoredQAgent.load(cfg.init_q_path, episode_len=max_steps)
        print(f"Checkpoint inicial cargado: {cfg.init_q_path}")
    elif os.path.exists(q_path):
        agent = FactoredQAgent.load(q_path, episode_len=max_steps)
        print(f"Checkpoint existente cargado: {q_path}")
    else:
        agent = FactoredQAgent(
            cfg=FactoredQConfig(alpha=cfg.alpha, gamma=cfg.gamma),
            encoder=FactoredStateEncoder(episode_len=max_steps),
            seed=cfg.base_seed,
        )

    eps_start = cfg.epsilon_start if cfg.epsilon_start is not None else agent.epsilon
    agent.epsilon = eps_start
    scheduler = _epsilon_scheduler(
        eps_start, cfg.epsilon_end, cfg.epsilon_decay_steps or n_episodes
    )

    def epsilon_for_episode(ep_num: int) -> float:
        """Epsilon al inicio del episodio (episodios 1-indexed)."""
        return scheduler(max(ep_num - 1, 0))

    t0 = time.time()
    last_rewards: deque[float] = deque(maxlen=50)
    last_deltas: deque[float] = deque(maxlen=50)
    q_usage_total = {"Q1": 0, "Q3": 0, "none": 0}
    global_updates = 0
    global_steps = 0
    snapshot_version = 0

    header = [
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
        "snapshot_version",
        "global_updates",
        "global_steps",
    ]

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with mp.Pool(processes=cfg.n_workers) as pool:
            pbar = tqdm(total=n_episodes, desc="Parallel Training", unit="ep")
            episode = 1
            while episode <= n_episodes:
                chunk_end = min(n_episodes, episode + cfg.sync_every - 1)
                snapshot = _snapshot_agent(agent)
                snapshot_version += 1

                tasks = {
                    ep: pool.apply_async(
                        _run_episode_worker,
                        (
                            ep,
                            base_cfg.__dict__,
                            snapshot,
                            epsilon_for_episode(ep),
                            max_steps,
                            cfg.base_seed,
                        ),
                    )
                    for ep in range(episode, chunk_end + 1)
                }

                buffer: Dict[int, EpisodeResult] = {}
                expected = episode

                while expected <= chunk_end:
                    for ep, task in list(tasks.items()):
                        if task.ready():
                            buffer[ep] = task.get()
                            del tasks[ep]

                    # Update progress bar postfix to show buffered episodes
                    running_count = len(tasks)
                    buffered_count = len(buffer)
                    pbar.set_postfix_str(
                        f"running={running_count} buffered={buffered_count} waiting_for=ep{expected}"
                    )

                    if expected in buffer:
                        result = buffer.pop(expected)
                        epsilon_ep = epsilon_for_episode(expected)
                        t_apply_start = time.time()
                        apply_stats = _apply_episode_to_agent(agent, result, epsilon_ep)
                        t_apply = time.time() - t_apply_start
                        agent.epsilon = epsilon_for_episode(expected + 1)

                        global_updates += len(result.transitions)
                        global_steps += result.steps
                        for k in q_usage_total:
                            q_usage_total[k] += result.q_usage.get(k, 0)

                        pending_avg = result.pending_sum / max(1, result.steps)
                        ticks_total = result.snap_end.get("t", result.steps)
                        wait_ratio = (
                            result.wait_count / result.action_count
                            if result.action_count
                            else 0.0
                        )
                        avg_rider_load = (
                            result.load_positive_sum / result.load_positive_count
                            if result.load_positive_count
                            else 0.0
                        )
                        batching_efficiency = (
                            result.ticks_with_batching / ticks_total
                            if ticks_total
                            else 0.0
                        )

                        max_delta = apply_stats["max_delta"]
                        last_rewards.append(result.reward)
                        last_deltas.append(max_delta)
                        reward_avg_50 = sum(last_rewards) / len(last_rewards)
                        delta_avg_50 = sum(last_deltas) / len(last_deltas)
                        elapsed = time.time() - t0

                        writer.writerow(
                            [
                                expected,
                                round(result.reward, 3),
                                round(reward_avg_50, 3),
                                round(pending_avg, 3),
                                result.delivered_total,
                                result.delivered_ontime,
                                result.delivered_late,
                                round(epsilon_ep, 4),
                                round(max_delta, 6),
                                round(delta_avg_50, 6),
                                round(elapsed, 2),
                                result.seed,
                                result.q_usage["Q1"],
                                result.q_usage["Q3"],
                                apply_stats["stats"]["Q1_entries"],
                                apply_stats["stats"]["Q3_entries"],
                                result.snap_end.get("t", result.steps),
                                result.wait_count,
                                result.action_count,
                                round(wait_ratio, 4),
                                round(avg_rider_load, 4),
                                round(batching_efficiency, 4),
                                result.activated_riders,
                                snapshot_version,
                                global_updates,
                                global_steps,
                            ]
                        )

                        if (expected % cfg.log_every) == 0:
                            pbar.write(
                                f"[Ep {expected}] r={result.reward:.2f} "
                                f"avg50={reward_avg_50:.2f} eps={epsilon_ep:.3f} "
                                f"delta_max={max_delta:.4f} wait_ratio={wait_ratio:.3f} "
                                f"apply_time={t_apply:.1f}s"
                            )
                            f.flush()

                        if cfg.eval_every and (expected % cfg.eval_every) == 0:
                            eval_summary = _evaluate_greedy(
                                agent, base_cfg, cfg.eval_episodes, cfg.base_seed
                            )
                            pbar.write(
                                f"  Eval@{expected}: reward_avg={eval_summary['reward_avg']:.2f}, "
                                f"wait_ratio={eval_summary['wait_ratio_avg']:.3f}, "
                                f"batching_eff={eval_summary['batching_eff_avg']:.3f}"
                            )

                        if (expected % cfg.save_every) == 0:
                            agent.save(q_path)
                            f.flush()

                        expected += 1
                        pbar.update(1)
                    else:
                        time.sleep(0.01)

                episode = chunk_end + 1
            pbar.close()

    agent.save(q_path)
    print(f"\nEntrenamiento paralelo finalizado. Q-table: {q_path}")
    print(f"Epsilon inicial={eps_start:.4f}, final={scheduler(n_episodes):.4f}")
    print(
        f"Episodios={n_episodes}, updates={global_updates}, steps={global_steps}"
    )
    print(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Entrenamiento Q-Learning factorizado (secuencial o paralelo)"
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
        help=f"Modo rápido: {FAST_EPISODES} episodios de {FAST_MAX_TICKS} ticks para smoke test",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Activar entrenamiento paralelo"
    )
    parser.add_argument("--n-workers", type=int, default=1, help="Número de workers")
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help="Máximo de steps por episodio (default=episode_len)",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=None,
        help="Epsilon inicial (None usa valor cargado o por defecto)",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Epsilon mínimo",
    )
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=0,
        help="Pasos de decaimiento (0 usa n_episodes)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Evaluar cada N episodios (0 deshabilita)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Número de episodios de evaluación",
    )
    parser.add_argument(
        "--log-every", type=int, default=10, help="Log por consola cada N episodios"
    )
    parser.add_argument(
        "--save-every", type=int, default=50, help="Guardar Q-table cada N episodios"
    )
    parser.add_argument(
        "--sync-every",
        type=int,
        default=10,
        help="Refrescar snapshot de política cada N episodios",
    )
    parser.add_argument(
        "--init-qpath",
        type=str,
        default=None,
        help="Checkpoint inicial para arranque",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="artifacts/metrics_factored_parallel.csv",
        help="Ruta del csv de métricas (paralelo)",
    )
    parser.add_argument(
        "--qpath",
        type=str,
        default="artifacts/qtable_factored.pkl",
        help="Ruta de Q-table a guardar/cargar",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Learning rate (alpha) para Q-learning",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Factor de descuento (gamma) para Q-learning",
    )
    args = parser.parse_args()

    if args.parallel:
        max_steps = args.max_steps_per_episode or args.episode_len
        parallel_cfg = ParallelTrainConfig(
            n_episodes=args.episodes,
            out_dir="artifacts",
            base_seed=args.seed,
            episode_len=args.episode_len,
            max_steps_per_episode=max_steps,
            n_workers=max(1, args.n_workers),
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            log_every=max(1, args.log_every),
            save_every=max(1, args.save_every),
            sync_every=max(1, args.sync_every),
            fast=args.fast,
            init_q_path=args.init_qpath,
            q_path=args.qpath,
            metrics_path=args.metrics_path,
            alpha=args.alpha,
            gamma=args.gamma,
        )
        train_parallel(parallel_cfg)
    else:
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
