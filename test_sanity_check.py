# test_sanity_check.py
"""
Experimento "Juguete" para verificar que el agente es capaz de aprender ALGO.
Grid pequeño, 1 rider, sin tráfico dinámico.

Si este test PASA:
- max_delta_q > 0 (hay actualizaciones)
- reward mejora con el tiempo
- ASSIGN_ANY se prefiere sobre WAIT

Si este test FALLA:
- Revisar que las correcciones estén bien aplicadas
"""
from __future__ import annotations
import os

from simulation.simulator import Simulator, SimConfig
from core.factored_states import FactoredStateEncoder
from core.factored_q_agent import FactoredQAgent, FactoredQConfig


def run_sanity_check(n_episodes: int = 50, verbose: bool = True):
    """Ejecuta el experimento de sanity check."""

    # Config minimalista
    cfg = SimConfig(
        width=7,  # Grid mínimo
        height=7,
        n_riders=1,  # UN solo rider
        episode_len=100,  # Episodios cortos
        order_spawn_prob=0.25,  # Pocos pedidos
        max_eta=50,  # Deadline generoso
        block_size=3,
        street_width=1,
        seed=42,
        enable_internal_spawn=True,
        enable_internal_traffic=False,  # SIN tráfico dinámico
    )

    agent = FactoredQAgent(
        cfg=FactoredQConfig(
            alpha=0.3,  # Learning rate moderado-alto
            gamma=0.9,
            eps_start=1.0,
            eps_min=0.1,
            eps_decay=0.95,  # Decay rápido para test corto
        ),
        encoder=FactoredStateEncoder(episode_len=cfg.episode_len),
        seed=42,
    )

    rewards = []
    deltas = []
    action_counts = {"Q1": 0, "Q3": 0, "none": 0}

    for ep in range(1, n_episodes + 1):
        # Seed distinta por episodio
        ep_cfg = SimConfig(**cfg.__dict__)
        ep_cfg.seed = 42 + ep

        sim = Simulator(ep_cfg)
        agent.encoder.reset()
        agent.reset_delta()

        total_r = 0.0
        ep_actions = {"ASSIGN_URGENT": 0, "ASSIGN_ANY": 0, "WAIT": 0, "REPLAN": 0}

        snap = sim.snapshot()
        done = False

        while not done:
            action = agent.choose_action(snap, training=True)
            action_counts[agent.last_q_used] += 1

            # Contar acciones
            if action == 0:
                ep_actions["ASSIGN_URGENT"] += 1
            elif action == 1:
                ep_actions["ASSIGN_ANY"] += 1
            elif action == 2:
                ep_actions["WAIT"] += 1
            elif action == 3:
                ep_actions["REPLAN"] += 1

            reward, done = sim.step(action)
            total_r += reward

            snap_next = sim.snapshot()
            agent.update(snap, action, reward, snap_next, done)
            agent.commit_encoder(snap)
            snap = snap_next

        agent.decay_epsilon()
        rewards.append(total_r)
        deltas.append(agent.get_max_delta())

        if verbose and ep % 10 == 0:
            avg_r = sum(rewards[-10:]) / min(10, len(rewards[-10:]))
            avg_d = sum(deltas[-10:]) / min(10, len(deltas[-10:]))
            print(
                f"Ep {ep:3d}: reward={total_r:7.1f}, avg_10={avg_r:7.1f}, "
                f"delta={agent.get_max_delta():.4f}, eps={agent.epsilon:.3f}"
            )

    # === Verificaciones ===
    print("\n" + "=" * 60)
    print("RESULTADOS SANITY CHECK")
    print("=" * 60)

    # 1. ¿Hay updates de Q-values?
    total_delta = sum(deltas)
    print(f"\n1. Delta Q total: {total_delta:.4f}")
    if total_delta > 0.1:
        print("   [OK] PASS: Hay actualizaciones de Q-values")
    else:
        print("   [X] FAIL: No hay actualizaciones significativas")

    # 2. ¿Mejora el reward?
    first_10 = sum(rewards[:10]) / 10
    last_10 = sum(rewards[-10:]) / 10
    improvement = last_10 - first_10
    print(f"\n2. Reward primeros 10: {first_10:.1f}")
    print(f"   Reward ultimos 10:  {last_10:.1f}")
    print(f"   Mejora: {improvement:+.1f}")
    if improvement > 0:
        print("   [OK] PASS: El reward mejora con el tiempo")
    else:
        print("   [!] WARN: El reward no mejora (puede necesitar mas episodios)")

    # 3. ¿Se usa Q1 más que none?
    print(f"\n3. Uso de Q-tables:")
    print(
        f"   Q1: {action_counts['Q1']} ({100*action_counts['Q1']/max(1,sum(action_counts.values())):.1f}%)"
    )
    print(
        f"   Q3: {action_counts['Q3']} ({100*action_counts['Q3']/max(1,sum(action_counts.values())):.1f}%)"
    )
    print(
        f"   none: {action_counts['none']} ({100*action_counts['none']/max(1,sum(action_counts.values())):.1f}%)"
    )
    if action_counts["Q1"] > action_counts["none"]:
        print("   [OK] PASS: Q1 se usa mas que idle")
    else:
        print("   [!] WARN: Demasiado tiempo idle")

    # 4. Stats del agente
    stats = agent.stats()
    print(f"\n4. Q-table stats:")
    print(f"   Q1 entries: {stats['Q1_entries']}")
    print(f"   Q3 entries: {stats['Q3_entries']}")
    print(f"   Epsilon final: {agent.epsilon:.4f}")

    return rewards, deltas, agent


if __name__ == "__main__":
    run_sanity_check(n_episodes=50, verbose=True)
