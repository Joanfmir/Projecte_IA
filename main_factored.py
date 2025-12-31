# main_factored.py
"""Script principal para ejecutar el agente factorizado.

Este script permite correr simulaciones utilizando un agente entrenado con Q-Learning factorizado (o una heurística).
Soporta dos modos de operación:
- Headless: Ejecución rápida en consola para obtener métricas.
- Visual: Interfaz gráfica (GUI) con Matplotlib para visualizar el comportamiento del agente.

Ejemplo de uso:
    python main_factored.py --visual --policy trained --qpath artifacts/qtable_factored.pkl
"""
from __future__ import annotations
import argparse
import os

from simulation.simulator import Simulator, SimConfig
from simulation.visualizer import Visualizer
from core.factored_states import FactoredStateEncoder
from core.factored_q_agent import FactoredQAgent
from core.dispatch_policy import A_ASSIGN_ANY_NEAREST


class TrainedFactoredPolicy:
    """Wrapper de política para usar un agente factorizado entrenado en la simulación.

    Adapta la interfaz del `FactoredQAgent` para que sea compatible con el bucle de
    simulación y el visualizador.

    Attributes:
        agent: Instancia cargada de `FactoredQAgent`.
        episode_len: Duración del episodio para normalización en el encoder.
    """

    def __init__(self, q_path: str, episode_len: int = 900):
        self.agent = FactoredQAgent.load(q_path, episode_len=episode_len)
        self.agent.epsilon = 0.0  # Greedy siempre (explotación pura)
        self.episode_len = episode_len

    def choose_action_snapshot(self, snap: dict) -> int:
        """Selecciona una acción basada en el snapshot del simulador.

        Args:
            snap: Diccionario con el estado actual de la simulación.

        Returns:
            ID de la acción seleccionada.
        """
        return self.agent.choose_action(snap, training=False)

    def reset(self):
        """Reinicia el estado interno del encoder al inicio de un episodio."""
        self.agent.encoder.reset()


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    p = argparse.ArgumentParser(description="Ejecutar agente factorizado")

    # Modo
    p.add_argument("--visual", action="store_true", help="Abrir simulación con GUI")
    p.add_argument("--policy", choices=["heuristic", "trained"], default="trained", help="Política a usar")

    # Paths
    p.add_argument("--qpath", default="artifacts/qtable_factored.pkl", help="Ruta a la tabla Q guardada")

    # Configuraciones del Simulador
    p.add_argument("--width", type=int, default=45, help="Ancho de la grilla")
    p.add_argument("--height", type=int, default=35, help="Alto de la grilla")
    p.add_argument("--riders", type=int, default=6, help="Número de repartidores")
    p.add_argument("--episode_len", type=int, default=900, help="Duración del episodio (ticks)")
    p.add_argument("--spawn", type=float, default=0.40, help="Probabilidad de generación de pedidos")
    p.add_argument("--max_eta", type=int, default=80, help="Tiempo máximo de entrega inicial")
    p.add_argument("--block_size", type=int, default=6, help="Tamaño de bloques de edificios")
    p.add_argument("--street_width", type=int, default=2, help="Ancho de calles")
    p.add_argument("--seed", type=int, default=7, help="Semilla aleatoria")

    # Visualización
    p.add_argument(
        "--interval_ms", type=int, default=200, help="Intervalo entre frames (ms)"
    )

    # Eventos dinámicos
    p.add_argument("--traffic-interval", type=int, default=60, help="Intervalo cambio de tráfico")
    p.add_argument("--closure-prob", type=float, default=0.02, help="Prob. cierre de calles")
    p.add_argument("--no-events", action="store_true", help="Desactivar eventos dinámicos")

    return p.parse_args()


def make_config(args) -> SimConfig:
    """Crea la configuración de la simulación basada en argumentos."""
    # Spawn interno ON (necesitamos pedidos), tráfico interno ON (mismo régimen que entrenamiento)
    return SimConfig(
        width=args.width,
        height=args.height,
        n_riders=args.riders,
        episode_len=args.episode_len,
        order_spawn_prob=args.spawn,
        max_eta=args.max_eta,
        seed=args.seed,
        block_size=args.block_size,
        street_width=args.street_width,
        enable_internal_spawn=True,
        enable_internal_traffic=True,  # Mismo régimen que entrenamiento
        road_closure_prob=args.closure_prob,  # Probabilidad de cierres de calles
    )


def run_headless(sim: Simulator, policy_name: str, qpath: str, args):
    """Ejecuta la simulación en modo consola (sin GUI).

    Muestra métricas periódicas y un resumen final.
    """
    import random

    rng = random.Random(args.seed)

    policy = None
    if policy_name == "trained":
        if not os.path.exists(qpath):
            raise FileNotFoundError(
                f"No existe {qpath}. Ejecuta: python train_factored.py"
            )
        policy = TrainedFactoredPolicy(qpath, episode_len=args.episode_len)
        policy.reset()

    done = False
    total_reward = 0.0

    print(f"Ejecutando episodio (seed={args.seed}, policy={policy_name})...")
    print()

    while not done:
        t = sim.t

        # Elegir acción
        snap = sim.snapshot()
        if policy_name == "heuristic":
            action = A_ASSIGN_ANY_NEAREST
        else:
            action = policy.choose_action_snapshot(snap)

        reward, done = sim.step(action)
        total_reward += reward

        # Commit encoder para que delta_traffic funcione en siguiente tick
        if policy_name == "trained" and policy is not None:
            policy.agent.commit_encoder(snap)

        # Progress cada 100 ticks
        if t % 100 == 0:
            print(
                f"  t={t:4d} | pending={len(snap['pending_orders']):3d} | "
                f"delivered={snap['delivered_total']:3d} | reward={total_reward:.1f}"
            )

    # Resultados finales
    end = sim.snapshot()
    print()
    print("=" * 50)
    print("RESULTADOS FINALES")
    print("=" * 50)
    print(f"  Total reward:      {total_reward:.1f}")
    print(f"  Orders created:    {len(sim.om.orders)}")
    print(f"  Orders delivered:  {end['delivered_total']}")
    print(f"  On-time:           {end['delivered_ontime']}")
    print(f"  Late:              {end['delivered_late']}")
    print(f"  Pending at end:    {len(end['pending_orders'])}")

    ontime_ratio = (
        end["delivered_ontime"] / end["delivered_total"]
        if end["delivered_total"] > 0
        else 0
    )
    print(f"  On-time ratio:     {ontime_ratio:.1%}")

    # Fatiga
    fatigues = [r["fatigue"] for r in end["riders"]]
    print(f"  Avg fatigue:       {sum(fatigues)/len(fatigues):.2f}")
    print(f"  Max fatigue:       {max(fatigues):.2f}")

    # Closures
    print(f"  Road closures:     {sim.graph.count_closed_directed()}")


def run_visual(sim: Simulator, policy_name: str, qpath: str, args):
    """Ejecuta la simulación con interfaz gráfica (GUI)."""
    import random

    rng = random.Random(args.seed)

    if policy_name == "heuristic":
        # Visualizer sin política usa heurística interna
        vis = Visualizer(sim, policy=None, interval_ms=args.interval_ms)
    else:
        if not os.path.exists(qpath):
            raise FileNotFoundError(
                f"No existe {qpath}. Ejecuta: python train_factored.py"
            )

        policy = TrainedFactoredPolicy(qpath, episode_len=args.episode_len)
        policy.reset()

        # El Visualizer necesita un objeto con choose_action_snapshot
        vis = Visualizer(sim, policy=policy, interval_ms=args.interval_ms)

    print(f"Iniciando visualización (seed={args.seed}, policy={policy_name})...")
    print("Cierra la ventana para terminar.")
    vis.run()


def main():
    args = parse_args()
    cfg = make_config(args)
    sim = Simulator(cfg)

    print(f"Simulator: {cfg.width}x{cfg.height}, {cfg.n_riders} riders")
    print(f"Episode length: {cfg.episode_len} ticks")
    print()

    if args.visual:
        run_visual(sim, args.policy, args.qpath, args)
    else:
        run_headless(sim, args.policy, args.qpath, args)


if __name__ == "__main__":
    main()
