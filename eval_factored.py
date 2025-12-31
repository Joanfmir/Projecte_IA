# eval_factored.py
"""Evaluación reproducible del agente Q-Learning factorizado.

Este script compara el rendimiento del agente entrenado (`FactoredQAgent`) contra
una política heurística baseline (asignar al más cercano).

Genera métricas detalladas incluyendo:
- Tasa de entregas a tiempo (On-time ratio).
- Recompensa total acumulada.
- Tiempos promedio de entrega.
- Fatiga de los repartidores y balance de carga.
- Uso de tablas Q (Q1 vs Q3).

Los resultados se guardan en JSON para posterior análisis.
"""
from __future__ import annotations
import argparse
import statistics
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from simulation.simulator import Simulator, SimConfig
from core.dispatch_policy import (
    A_ASSIGN_ANY_NEAREST,
    A_ASSIGN_URGENT_NEAREST,
    A_WAIT,
    A_REPLAN_TRAFFIC,
)
from core.factored_q_agent import FactoredQAgent

FAST_EPISODES = 2
FAST_MAX_TICKS = 200


@dataclass
@dataclass
class EvalConfig:
    """Configuración de los parámetros de evaluación."""

    n_episodes: int = 20
    episode_len: int = 900

    # Simulador
    width: int = 45
    height: int = 35
    n_riders: int = 6
    order_spawn_prob: float = 0.40
    max_eta: int = 80
    block_size: int = 6
    street_width: int = 2

    # Eventos dinámicos
    # Tráfico manejado internamente por simulator (enable_internal_traffic=True)
    enable_road_closures: bool = False
    road_closure_prob: float = 0.02  # probabilidad por tick de nuevo cierre
    max_closures: int = 5  # máximo de cierres activos

    # Reproducibilidad
    base_seed: Optional[int] = 7  # None = aleatorio

    # Paths
    q_path: str = "artifacts/qtable_factored.pkl"


@dataclass
class EpisodeMetrics:
    """Contenedor de métricas para un episodio individual."""

    seed: int
    reward_total: float

    # Pedidos
    orders_created: int
    orders_delivered: int
    orders_ontime: int
    orders_late: int
    orders_pending_end: int
    ontime_ratio: float

    # Tiempos
    avg_delivery_time: float

    # Riders
    avg_fatigue: float
    max_fatigue: float
    fatigue_std: float
    avg_distance: float
    deliveries_imbalance: float  # std de entregas por rider

    # Tráfico
    traffic_changes: int
    road_closures_total: int

    # Q-table usage (solo para agente)
    q1_used: int = 0
    q3_used: int = 0
    ticks_total: int = 0
    wait_count: int = 0
    action_count: int = 0
    wait_ratio: float = 0.0
    avg_rider_load: float = 0.0
    batching_efficiency: float = 0.0
    activated_riders: int = 0


def run_episode_with_events(
    sim: Simulator,
    policy_fn,
    cfg: EvalConfig,
    track_q_usage: bool = False,
    agent: "FactoredQAgent" = None,  # Para llamar commit_encoder
) -> EpisodeMetrics:
    """Ejecuta un episodio de evaluación con eventos dinámicos configurados.

    Args:
        sim: Instancia del simulador ya configurada.
        policy_fn: Función que toma (sim, snap) y devuelve (acción, q_used).
        cfg: Configuración de evaluación.
        track_q_usage: Si es True, cuenta cuántas veces se usó Q1/Q3.
        agent: Referencia al agente (opcional) para actualizar su encoder.

    Returns:
        Objeto `EpisodeMetrics` con los resultados del episodio.
    """
    rng = sim.rng

    total_r = 0.0
    traffic_changes = 0
    road_closures_total = 0
    q_usage = {"Q1": 0, "Q3": 0, "none": 0}
    wait_count = 0
    action_count = 0
    load_positive_sum = 0
    load_positive_count = 0
    ticks_with_batching = 0
    activated_riders = set()

    # Tracking de tiempos de entrega
    delivery_times: List[int] = []

    snapshot_fn = sim.snapshot
    step_fn = sim.step

    done = False
    prev_traffic_zones = dict(getattr(sim, "traffic_zones", {}))

    while not done:

        # === Eventos dinámicos ===
        # Tráfico manejado internamente por simulator.step() (enable_internal_traffic=True)

        # Detectar cambios de tráfico (para métricas)
        current_zones = dict(getattr(sim, "traffic_zones", {}))
        if current_zones != prev_traffic_zones:
            traffic_changes += 1
            prev_traffic_zones = current_zones

        # Cierres de calles aleatorios (estos SÍ se inyectan externamente)
        if cfg.enable_road_closures and rng.random() < cfg.road_closure_prob:
            current_closures = sim.graph.count_closed_directed()
            if current_closures < cfg.max_closures * 2:  # *2 porque son bidireccionales
                sim.graph.random_road_incidents(1)
                road_closures_total += 1

        # === Política ===
        snap = snapshot_fn()

        # Métricas de batching
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

        action, q_used = policy_fn(sim, snap)

        if track_q_usage and q_used:
            q_usage[q_used] += 1
        if action == A_WAIT:
            wait_count += 1
        action_count += 1

        # === Step ===
        # Guardamos pedidos antes del step para detectar entregas
        pending_before = set(o.order_id for o in sim.om.get_pending_orders())

        reward, done = step_fn(action)
        total_r += reward

        # CRÍTICO: Actualizar prev_traffic_pressure para que delta_traffic funcione
        if agent is not None:
            agent.commit_encoder(snap)

        # Detectar entregas y calcular tiempo
        pending_after = set(o.order_id for o in sim.om.get_pending_orders())
        delivered_this_step = pending_before - pending_after

        for oid in delivered_this_step:
            for o in sim.om.orders:
                if o.order_id == oid and o.delivered_at is not None:
                    delivery_times.append(o.delivered_at - o.created_at)

    # === Calcular métricas finales ===
    snap_end = sim.snapshot()
    riders = snap_end["riders"]

    fatigues = [r["fatigue"] for r in riders]
    distances = [r.get("distance", 0) for r in riders]

    # Contar entregas por rider
    deliveries_per_rider = []
    for r in sim.fm.get_all():
        deliveries_per_rider.append(r.deliveries_done)

    if len(deliveries_per_rider) >= 2:
        mean_del = sum(deliveries_per_rider) / len(deliveries_per_rider)
        var_del = sum((x - mean_del) ** 2 for x in deliveries_per_rider) / len(
            deliveries_per_rider
        )
        del_imbalance = var_del**0.5
    else:
        del_imbalance = 0.0

    delivered_total = snap_end.get("delivered_total", 0)
    ontime = snap_end.get("delivered_ontime", 0)
    late = snap_end.get("delivered_late", 0)
    ticks_total = snap_end.get("t", 0)
    wait_ratio = wait_count / action_count if action_count else 0.0
    batching_efficiency = (
        ticks_with_batching / ticks_total if ticks_total else 0.0
    )

    return EpisodeMetrics(
        seed=sim.cfg.seed,
        reward_total=total_r,
        orders_created=len(sim.om.orders),
        orders_delivered=delivered_total,
        orders_ontime=ontime,
        orders_late=late,
        orders_pending_end=len(snap_end["pending_orders"]),
        ontime_ratio=ontime / delivered_total if delivered_total > 0 else 0,
        avg_delivery_time=(
            sum(delivery_times) / len(delivery_times) if delivery_times else 0
        ),
        avg_fatigue=sum(fatigues) / len(fatigues) if fatigues else 0,
        max_fatigue=max(fatigues) if fatigues else 0,
        fatigue_std=statistics.pstdev(fatigues) if len(fatigues) >= 2 else 0,
        avg_distance=sum(distances) / len(distances) if distances else 0,
        deliveries_imbalance=del_imbalance,
        traffic_changes=traffic_changes,
        road_closures_total=road_closures_total,
        q1_used=q_usage["Q1"],
        q3_used=q_usage["Q3"],
        ticks_total=ticks_total,
        wait_count=wait_count,
        action_count=action_count,
        wait_ratio=wait_ratio,
        avg_rider_load=(
            load_positive_sum / load_positive_count if load_positive_count else 0.0
        ),
        batching_efficiency=batching_efficiency,
        activated_riders=len(activated_riders),
    )


def make_heuristic_policy():
    """Crea una política heurística baseline: siempre asignar al rider más cercano.

    Returns:
        Una función `policy(sim, snap)` que devuelve la acción `A_ASSIGN_ANY_NEAREST`.
    """

    def policy(sim, snap):
        return A_ASSIGN_ANY_NEAREST, None

    return policy


def make_factored_policy(q_path: str, episode_len: int):
    """Carga un agente factorizado y crea una función de política compatible.

    Args:
        q_path: Ruta al archivo pickle de la Q-table.
        episode_len: Duración del episodio para el encoder.

    Returns:
        Tupla (policy_fn, agent_instance, prev_epsilon).
    """
    agent = FactoredQAgent.load(q_path, episode_len=episode_len)
    prev_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Greedy en evaluación

    def policy(sim, snap):
        action = agent.choose_action(snap, training=False)
        return action, agent.last_q_used

    return policy, agent, prev_epsilon


def evaluate(cfg: EvalConfig):
    """Ejecuta la evaluación completa comparando Heurística vs Agente Factorizado.

    Imprime resultados en consola y guarda un archivo JSON detallado.
    """

    # Determinar seed base
    if cfg.base_seed is None:
        cfg.base_seed = random.randint(1, 100000)

    print(f"=== Evaluación Reproducible ===")
    print(f"Base seed: {cfg.base_seed}")
    print(f"Episodios: {cfg.n_episodes}")
    print(f"Traffic: interno (enable_internal_traffic=True)")
    print(f"Road closures: {'ON' if cfg.enable_road_closures else 'OFF'}")
    print()

    # Crear políticas
    heuristic_policy = make_heuristic_policy()
    agent_prev_epsilon: Optional[float] = None

    try:
        factored_policy, agent, agent_prev_epsilon = make_factored_policy(
            cfg.q_path, cfg.episode_len
        )
        has_trained = True
        print(f"Agente cargado: {agent.stats()}")
    except FileNotFoundError:
        print(f"⚠️  No se encontró {cfg.q_path}. Solo se evaluará heurística.")
        has_trained = False

    print()

    # Resultados
    results_h: List[EpisodeMetrics] = []
    results_q: List[EpisodeMetrics] = []

    for i in range(cfg.n_episodes):
        ep_seed = cfg.base_seed + i

        # Config del simulador
        # Spawn interno ON (necesitamos pedidos), tráfico interno ON (mismo régimen que train)
        sim_cfg = SimConfig(
            width=cfg.width,
            height=cfg.height,
            n_riders=cfg.n_riders,
            episode_len=cfg.episode_len,
            order_spawn_prob=cfg.order_spawn_prob,
            max_eta=cfg.max_eta,
            block_size=cfg.block_size,
            street_width=cfg.street_width,
            seed=ep_seed,
            enable_internal_spawn=True,
            enable_internal_traffic=True,  # MISMO régimen que entrenamiento
        )

        # Evaluar heurística
        sim_h = Simulator(sim_cfg)
        try:
            metrics_h = run_episode_with_events(
                sim_h, heuristic_policy, cfg, track_q_usage=False
            )
        finally:
            del sim_h
        results_h.append(metrics_h)

        # Evaluar agente factorizado
        if has_trained:
            sim_q = Simulator(sim_cfg)
            agent.encoder.reset()
            try:
                metrics_q = run_episode_with_events(
                    sim_q, factored_policy, cfg, track_q_usage=True, agent=agent
                )
            finally:
                del sim_q
            results_q.append(metrics_q)

        print(
            f"Ep {i+1}/{cfg.n_episodes} [seed={ep_seed}] - H: {metrics_h.ontime_ratio:.2%} on-time",
            end="",
        )
        if has_trained:
            print(f" | Q: {metrics_q.ontime_ratio:.2%} on-time")
        else:
            print()

    # === Resumen ===
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)

    def summarize(name: str, results: List[EpisodeMetrics]):
        print(f"\n--- {name} ---")

        def avg(key):
            vals = [getattr(m, key) for m in results]
            return sum(vals) / len(vals)

        def std(key):
            vals = [getattr(m, key) for m in results]
            return statistics.pstdev(vals) if len(vals) >= 2 else 0

        print(
            f"  Reward:           {avg('reward_total'):>10.1f} ± {std('reward_total'):.1f}"
        )
        print(f"  Orders created:   {avg('orders_created'):>10.1f}")
        print(f"  Orders delivered: {avg('orders_delivered'):>10.1f}")
        print(f"  On-time ratio:    {avg('ontime_ratio')*100:>10.1f}%")
        print(f"  Avg delivery time:{avg('avg_delivery_time'):>10.1f} ticks")
        print(f"  Avg fatigue:      {avg('avg_fatigue'):>10.2f}")
        print(f"  Max fatigue:      {avg('max_fatigue'):>10.2f}")
        print(f"  Delivery imbal.:  {avg('deliveries_imbalance'):>10.2f}")
        print(f"  Traffic changes:  {avg('traffic_changes'):>10.1f}")
        print(f"  Road closures:    {avg('road_closures_total'):>10.1f}")

        if results[0].q1_used > 0:
            print(f"  Q1 used (avg):    {avg('q1_used'):>10.1f}")
            print(f"  Q3 used (avg):    {avg('q3_used'):>10.1f}")

    summarize("HEURISTIC (nearest)", results_h)
    if has_trained:
        summarize("Q-LEARNING FACTORIZADO", results_q)

        # Comparación directa
        print("\n--- COMPARACIÓN ---")
        h_ontime = sum(m.ontime_ratio for m in results_h) / len(results_h)
        q_ontime = sum(m.ontime_ratio for m in results_q) / len(results_q)
        diff = (q_ontime - h_ontime) * 100
        print(
            f"  Diferencia on-time: {diff:+.1f}% ({'mejor' if diff > 0 else 'peor'} Q-Learning)"
        )

        h_reward = sum(m.reward_total for m in results_h) / len(results_h)
        q_reward = sum(m.reward_total for m in results_q) / len(results_q)
        print(f"  Diferencia reward:  {q_reward - h_reward:+.1f}")

        if agent_prev_epsilon is not None:
            agent.epsilon = agent_prev_epsilon

    # Guardar resultados
    output = {
        "config": asdict(cfg),
        "heuristic": [asdict(m) for m in results_h],
    }
    if has_trained:
        output["factored"] = [asdict(m) for m in results_q]

    output_path = "artifacts/eval_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResultados guardados en: {output_path}")

    return results_h, results_q if has_trained else None


def main():
    parser = argparse.ArgumentParser(description="Evaluación del agente factorizado")
    parser.add_argument("--episodes", type=int, default=20, help="Número de episodios")
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed base (por defecto 7; usa -1 para aleatorio)",
    )
    parser.add_argument("--qpath", type=str, default="artifacts/qtable_factored.pkl")
    parser.add_argument(
        "--no-closures", action="store_true", help="Deshabilitar cierres de calles"
    )
    parser.add_argument(
        "--closures",
        action="store_true",
        help="Habilitar cierres de calles (por defecto OFF para replicar train)",
    )
    parser.add_argument("--closure-prob", type=float, default=0.02)
    parser.add_argument(
        "--fast",
        "--debug",
        action="store_true",
        help=f"Modo rápido: {FAST_EPISODES} episodios de {FAST_MAX_TICKS} ticks para smoke test",
    )
    args = parser.parse_args()

    enable_road_closures = args.closures and not args.no_closures
    # --closures habilita, --no-closures fuerza OFF

    base_seed = args.seed
    if base_seed is not None and base_seed < 0:
        base_seed = None

    cfg = EvalConfig(
        n_episodes=args.episodes,
        base_seed=base_seed,
        q_path=args.qpath,
        enable_road_closures=enable_road_closures,
        road_closure_prob=args.closure_prob,
    )

    if args.fast:
        cfg.n_episodes = min(cfg.n_episodes, FAST_EPISODES)
        cfg.episode_len = min(cfg.episode_len, FAST_MAX_TICKS)

    evaluate(cfg)


if __name__ == "__main__":
    main()
