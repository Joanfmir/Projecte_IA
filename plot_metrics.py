# plot_metrics.py
from __future__ import annotations
import os
import csv
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt


METRICS_PATH = "artifacts/metrics.csv"



def moving_average(x: List[float], window: int) -> np.ndarray:
    if window <= 1:
        return np.array(x, dtype=float)
    arr = np.array(x, dtype=float)
    if len(arr) < window:
        # si hay pocos puntos, devolvemos media simple
        return np.array([arr.mean()] * len(arr))
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def read_metrics(path: str) -> Dict[str, List]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe {path}. ¿Seguro que está en la raíz del proyecto?")

    cols = {
        "episode": [],
        "reward": [],
        "pending_avg": [],
        "delivered_total": [],
        "ontime": [],
        "late": [],
        "epsilon": [],
        "sec_elapsed": [],
    }

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cols["episode"].append(int(row["episode"]))
            cols["reward"].append(float(row["reward"]))
            cols["pending_avg"].append(float(row.get("pending_avg", 0.0)))
            cols["delivered_total"].append(int(float(row.get("delivered_total", 0))))
            cols["ontime"].append(int(float(row.get("ontime", 0))))
            cols["late"].append(int(float(row.get("late", 0))))
            cols["epsilon"].append(float(row.get("epsilon", 0.0)))
            cols["sec_elapsed"].append(float(row.get("sec_elapsed", 0.0)))
    return cols


def save_plot(fig, filename: str):
    os.makedirs("plots", exist_ok=True)
    out = os.path.join("plots", filename)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("Guardado:", out)


def plot_reward(data: Dict[str, List], window: int = 25):
    ep = data["episode"]
    r = data["reward"]

    fig = plt.figure(figsize=(9, 4.5))
    plt.plot(ep, r, linewidth=1, alpha=0.55, label="Reward (episodio)")

    sm = moving_average(r, window=window)
    # para alinear con episodios (valid reduce longitud)
    ep_sm = ep[len(ep) - len(sm):]
    plt.plot(ep_sm, sm, linewidth=2, label=f"Media móvil (window={window})")

    # Comparación inicio vs final (si hay suficientes episodios)
    n = len(r)
    k = min(100, n // 3) if n >= 30 else max(1, n // 2)
    first_mean = np.mean(r[:k])
    last_mean = np.mean(r[-k:])
    plt.axhline(first_mean, linestyle="--", linewidth=1, label=f"Media primeros {k}: {first_mean:.0f}")
    plt.axhline(last_mean, linestyle="--", linewidth=1, label=f"Media últimos {k}: {last_mean:.0f}")

    plt.title("Evolución del reward durante el entrenamiento")
    plt.xlabel("Episodio")
    plt.ylabel("Reward total")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    save_plot(fig, "reward.png")
    return fig


def plot_pending(data: Dict[str, List], window: int = 25):
    ep = data["episode"]
    p = data["pending_avg"]

    fig = plt.figure(figsize=(9, 4.5))
    plt.plot(ep, p, linewidth=1, alpha=0.6, label="Pending avg (episodio)")

    sm = moving_average(p, window=window)
    ep_sm = ep[len(ep) - len(sm):]
    plt.plot(ep_sm, sm, linewidth=2, label=f"Media móvil (window={window})")

    plt.title("Pedidos pendientes (media) durante el entrenamiento")
    plt.xlabel("Episodio")
    plt.ylabel("Pending avg")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    save_plot(fig, "pending_avg.png")
    return fig


def plot_ontime_late(data: Dict[str, List], window: int = 25):
    ep = data["episode"]
    on = data["ontime"]
    late = data["late"]

    fig = plt.figure(figsize=(9, 4.5))
    plt.plot(ep, on, linewidth=1, alpha=0.55, label="On-time (episodio)")
    plt.plot(ep, late, linewidth=1, alpha=0.55, label="Late (episodio)")

    on_sm = moving_average(on, window=window)
    late_sm = moving_average(late, window=window)
    ep_sm = ep[len(ep) - len(on_sm):]
    plt.plot(ep_sm, on_sm, linewidth=2, label=f"On-time media móvil (w={window})")
    plt.plot(ep_sm, late_sm, linewidth=2, label=f"Late media móvil (w={window})")

    plt.title("Entregas a tiempo vs tarde (por episodio)")
    plt.xlabel("Episodio")
    plt.ylabel("Nº entregas")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    save_plot(fig, "ontime_late.png")
    return fig


def plot_epsilon(data: Dict[str, List]):
    ep = data["episode"]
    eps = data["epsilon"]

    fig = plt.figure(figsize=(9, 4.5))
    plt.plot(ep, eps, linewidth=2)
    plt.title("Epsilon (exploración) durante el entrenamiento")
    plt.xlabel("Episodio")
    plt.ylabel("Epsilon")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    save_plot(fig, "epsilon.png")
    return fig


def main():
    data = read_metrics(METRICS_PATH)

    # Ventana de media móvil: ajusta si quieres más suavizado
    window = 25

    fig1 = plot_reward(data, window=window)
    fig2 = plot_pending(data, window=window)
    fig3 = plot_ontime_late(data, window=window)
    fig4 = plot_epsilon(data)

    plt.show()


if __name__ == "__main__":
    main()
