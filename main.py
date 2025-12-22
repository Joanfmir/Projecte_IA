# main.py
# =========================
# Pizza Delivery IA - MAIN
# Versión realista: calles + edificios (obstáculos)
# =========================

import matplotlib
# En algunos Macs evita problemas con la ventana
matplotlib.use("TkAgg")

from simulation.simulator import Simulator, SimConfig
from simulation.visualizer import Visualizer


def run_visual():
    """
    Lanza la simulación visual:
    - Ciudad con manzanas (edificios) y calles
    - Riders SOLO circulan por calles
    - Pedidos aparecen en celdas caminables
    """

    cfg = SimConfig(
        width=45,
        height=35,
        n_riders=4,
        episode_len=900,
        order_spawn_prob=0.1,
        max_eta=80,
        block_size=6,
        street_width=2,
        seed=7,
    )

    sim = Simulator(cfg)

    # policy=None -> heurística simple (nearest)
    vis = Visualizer(sim, policy=None, interval_ms=240)
    vis.run()


if __name__ == "__main__":
    run_visual()
