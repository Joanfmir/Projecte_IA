from __future__ import annotations

import random

from core.shared_params import A_ASSIGN_ANY_NEAREST, A_WAIT
from core.factored_q_agent import FactoredQAgent, FactoredQConfig
from core.factored_states import FactoredStateEncoder
from simulation.simulator import SimConfig, Simulator


def _base_agent(cfg: SimConfig, seed: int = 123) -> FactoredQAgent:
    return FactoredQAgent(
        cfg=FactoredQConfig(
            alpha=0.2,
            gamma=0.9,
            eps_start=0.0,
            eps_min=0.0,
            eps_decay=1.0,
        ),
        encoder=FactoredStateEncoder(episode_len=cfg.episode_len),
        seed=seed,
    )


def test_batching_prefers_partial_rider_cluster():
    random.seed(99)
    cfg = SimConfig(
        width=15,
        height=15,
        n_riders=3,
        episode_len=20,
        order_spawn_prob=0.0,
        max_eta=80,
        enable_internal_spawn=False,
        enable_internal_traffic=False,
        seed=77,
    )
    sim = Simulator(cfg)
    # Create 3 clustered orders with enough slack
    neighbors = [nb for nb, _ in sim.graph.neighbors(sim.restaurant)]
    drops = neighbors[:3]
    if len(drops) < 3:
        # Fallback determinista: reutilizar vecinos existentes o restaurante
        if not neighbors:
            drops = [sim.restaurant] * 3
        else:
            idx = 0
            while len(drops) < 3:
                drops.append(neighbors[idx % len(neighbors)])
                idx += 1
    for drop in drops:
        assert drop is not None
        sim.om.create_order(
            pickup=sim.restaurant,
            dropoff=drop,
            now=0,
            max_eta=60,
            priority=1,
        )

    agent = _base_agent(cfg, seed=99)

    snap = sim.snapshot()
    # Ejecutar 3 decisiones deterministas (greedy)
    for _ in range(3):
        # Asignación greedy explícita (probamos la heurística de batching, no la política aprendida)
        action = A_ASSIGN_ANY_NEAREST
        reward, done = sim.step(action)
        snap_next = sim.snapshot()
        agent.update(snap, action, reward, snap_next, done)
        agent.commit_encoder(snap)
        snap = snap_next

    assigned_riders = {
        o.assigned_to for o in sim.om.orders if o.assigned_to is not None
    }
    assert all(o.assigned_to is not None for o in sim.om.orders)
    # Todas las asignaciones deben recaer en un solo rider (batching)
    assert len(assigned_riders) <= 1


def test_wait_updates_q_value_on_backlog():
    random.seed(101)
    cfg = SimConfig(
        width=10,
        height=10,
        n_riders=1,
        episode_len=10,
        order_spawn_prob=0.0,
        enable_internal_spawn=False,
        enable_internal_traffic=False,
        seed=101,
    )
    sim = Simulator(cfg)
    sim.om.create_order(
        pickup=sim.restaurant,
        dropoff=(sim.restaurant[0] + 1, sim.restaurant[1]),
        now=0,
        max_eta=8,
        priority=2,  # urgent
    )

    agent = _base_agent(cfg, seed=101)
    snap = sim.snapshot()

    # Inicializar contexto de tabla (Q1) y luego forzar WAIT
    _ = agent.choose_action(snap, training=False)
    action = A_WAIT
    reward, done = sim.step(action)
    snap_next = sim.snapshot()

    encoded = agent.encoder.encode_all(snap)
    agent.update(snap, action, reward, snap_next, done)

    q_wait = agent.get_q(agent.Q1, encoded["s_assign"], A_WAIT)
    # El valor debe reflejar el coste de esperar en backlog crítico.
    # Asume recompensa negativa por penalización de pedidos urgentes sin asignar;
    # si la estructura de recompensas cambia, actualizar esta aserción.
    assert q_wait < 0
