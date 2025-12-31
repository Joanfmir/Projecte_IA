# plot_metrics.py
"""Script de utilidad para visualizar métricas de entrenamiento.

Lee un archivo CSV generado por `train.py` o `train_factored.py` y genera
gráficas PNG estáticas para análisis básico.
"""
from __future__ import annotations

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def read_metrics(path: str) -> pd.DataFrame:
    """Lee el CSV de métricas y valida su contenido básico."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero train.py o pasa la ruta correcta con --metrics"
        )
    df = pd.read_csv(path)
    if "episode" not in df.columns:
        raise ValueError("metrics.csv debe tener columna 'episode'")
    return df


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def save_plot(fig, out_dir: str, name: str, tag: str):
    """Guarda la figura actual en disco y la cierra."""
    ensure_dir(out_dir)
    base = f"{name}.png" if not tag else f"{name}_{tag}.png"
    path = os.path.join(out_dir, base)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def plot_series(df: pd.DataFrame, x: str, y: str, title: str, out_dir: str, name: str, tag: str):
    """Genera y guarda un gráfico de línea simple."""
    fig = plt.figure()
    plt.plot(df[x], df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    save_plot(fig, out_dir, name, tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="artifacts/metrics.csv", help="Ruta a metrics.csv")
    ap.add_argument("--out", default="plots", help="Carpeta de salida para los png")
    ap.add_argument("--tag", default="", help="Sufijo para no sobreescribir (ej: run_2025_12_24)")
    args = ap.parse_args()

    df = read_metrics(args.metrics)

    # Si no existe reward_avg_50 (runs viejos), lo calculamos
    if "reward_avg_50" not in df.columns and "reward" in df.columns:
        df["reward_avg_50"] = df["reward"].rolling(50, min_periods=1).mean()

    # Plots básicos (según columnas disponibles)
    if "reward" in df.columns:
        plot_series(df, "episode", "reward", "Reward por episodio", args.out, "reward", args.tag)

    if "reward_avg_50" in df.columns:
        plot_series(df, "episode", "reward_avg_50", "Reward (media móvil 50)", args.out, "reward_avg_50", args.tag)

    if "pending_avg" in df.columns:
        plot_series(df, "episode", "pending_avg", "Pedidos pendientes (avg)", args.out, "pending_avg", args.tag)

    if "delivered_total" in df.columns:
        plot_series(df, "episode", "delivered_total", "Entregas totales por episodio", args.out, "delivered_total", args.tag)

    if "ontime" in df.columns:
        plot_series(df, "episode", "ontime", "Entregas a tiempo", args.out, "ontime", args.tag)

    if "late" in df.columns:
        plot_series(df, "episode", "late", "Entregas tarde", args.out, "late", args.tag)

    if "epsilon" in df.columns:
        plot_series(df, "episode", "epsilon", "Epsilon (exploración)", args.out, "epsilon", args.tag)

    print("\nOK. Si sigues viendo plots antiguos, usa --tag para generar nombres nuevos.")


if __name__ == "__main__":
    main()
