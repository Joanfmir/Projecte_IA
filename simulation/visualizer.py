# simulation/visualizer.py
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from core.dispatch_policy import A_ASSIGN_ANY_NEAREST


class Visualizer:
    def __init__(self, sim, policy=None, interval_ms: int = 220):
        self.sim = sim
        self.policy = policy
        self.interval_ms = interval_ms


        snap = self.sim.snapshot()
        self.W = snap["width"]
        self.H = snap["height"]
        self.buildings = set(tuple(x) for x in snap.get("buildings", []))
        self.avenues = snap.get("avenues", [])

        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(top=0.88, right=0.78)  # espacio para leyenda

        self.ax.set_xlim(-0.5, self.W - 0.5)
        self.ax.set_ylim(-0.5, self.H - 0.5)
        self.ax.set_aspect("equal")

        # Calles: fondo claro
        self.ax.set_facecolor("#f5f5f5")

        # Grid suave
        self.ax.set_xticks(range(self.W))
        self.ax.set_yticks(range(self.H))
        self.ax.grid(True, linewidth=1.0, alpha=0.10)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.tick_params(length=0)

        # Edificios (sólidos)
        self._draw_buildings()
        self._draw_avenues()


        self.ax.set_title("Pizza Delivery IA", fontsize=18, pad=8)
        self.hud = self.fig.text(
            0.02, 0.96, "",
            fontsize=11, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.75, edgecolor="none")
        )

        # Capas dinámicas (fijamos colores para que la leyenda tenga sentido)
        self.shop = self.ax.scatter([], [], s=320, marker="s", color="#ff7f0e")          # naranja
        self.orders_normal = self.ax.scatter([], [], s=90, marker="o", color="#2ca02c")  # verde
        self.orders_urgent = self.ax.scatter([], [], s=160, marker="o", color="#d62728") # rojo

        self.rider_scatters = []
        self.route_lines = []
        self.rider_labels = []

        # ✅ construir leyenda UNA VEZ
        self._build_legend()
    def _draw_avenues(self):

        W, H = self.W, self.H

        for av in self.avenues:
            m = av["m"]
            b = av["b"]
            w = int(av.get("w", 1))

            xs, ys = [], []
            for x in range(W):
                y = m * x + b
                if 0 <= y <= H - 1:
                    xs.append(x)
                    ys.append(y)

            if len(xs) >= 2:
                # línea principal
                self.ax.plot(xs, ys, linewidth=3.0, alpha=0.25)

                # "ancho" de avenida: líneas paralelas
                for k in range(1, w + 1):
                    ys_up = [yy + k for yy in ys]
                    ys_dn = [yy - k for yy in ys]
                    self.ax.plot(xs, ys_up, linewidth=2.0, alpha=0.12)
                    self.ax.plot(xs, ys_dn, linewidth=2.0, alpha=0.12)


    def _build_legend(self):
        handles = [
            Line2D([0], [0], marker="s", linestyle="None", markersize=11,
                   markerfacecolor="#ff7f0e", markeredgecolor="none", label="Restaurante"),
            Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                   markerfacecolor="#2ca02c", markeredgecolor="none", label="Pedido (normal)"),
            Line2D([0], [0], marker="o", linestyle="None", markersize=10,
                   markerfacecolor="#d62728", markeredgecolor="none", label="Pedido (urgente)"),
            Line2D([0], [0], marker="^", linestyle="None", markersize=10,
                   markerfacecolor="#1f77b4", markeredgecolor="none", label="Repartidor"),
            Line2D([0], [0], linewidth=2, color="#1f77b4", label="Ruta"),
            Rectangle((0, 0), 1, 1, facecolor="#273043", alpha=0.35, label="Edificio / manzana"),
        ]

        # guardamos la referencia para que no “desaparezca”
        self.legend = self.ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            title="Leyenda",
            borderaxespad=0.0
        )

    def _draw_buildings(self):
        for (x, y) in self.buildings:
            rect = Rectangle((x - 0.5, y - 0.5), 1, 1,
                             linewidth=0, facecolor="#273043", alpha=0.35)
            self.ax.add_patch(rect)

    def _ensure_riders(self, n: int):
        # colores distintos por rider (Matplotlib lo elige si no fijas)
        while len(self.rider_scatters) < n:
            sc = self.ax.scatter([], [], s=170, marker="^", color="#1f77b4")
            self.rider_scatters.append(sc)
            (ln,) = self.ax.plot([], [], linewidth=2, color="#1f77b4")
            self.route_lines.append(ln)
            txt = self.ax.text(0, 0, "", fontsize=9, ha="left", va="bottom")
            self.rider_labels.append(txt)

    def _safe_offsets(self, pts):
        if not pts:
            return [[-1000, -1000]]
        return pts

    def _update(self, _frame):
        if self.policy is not None:
            s = self.sim.compute_state()
            a = self.policy.choose_action(s)
        else:
            a = A_ASSIGN_ANY_NEAREST

        _, done = self.sim.step(a)

        snap = self.sim.snapshot()
        t = snap["t"]
        orders = snap["pending_orders"]
        riders = snap["riders"]
        traffic = snap["traffic"]
        closures = snap["closures"]
        blocked = snap.get("blocked", 0)

        self.hud.set_text(
            f"t={t}   pending={len(orders)}   riders={len(riders)}   traffic={traffic}   closures={closures}   blocked={blocked}"
        )

        sx, sy = snap["restaurant"]
        self.shop.set_offsets([[sx, sy]])

        normal_pts, urgent_pts = [], []
        for (loc, priority, *_rest) in orders:
            (urgent_pts if priority > 1 else normal_pts).append([loc[0], loc[1]])

        self.orders_normal.set_offsets(self._safe_offsets(normal_pts))
        self.orders_urgent.set_offsets(self._safe_offsets(urgent_pts))

        self._ensure_riders(len(riders))
        for i, r in enumerate(riders):
            x, y = r["pos"]
            self.rider_scatters[i].set_offsets([[x, y]])

            carrying = r["carrying"]
            lab = f"R{r['id']}" if carrying is None else f"R{r['id']}→{carrying}"
            self.rider_labels[i].set_position((x + 0.12, y + 0.12))
            self.rider_labels[i].set_text(lab)

            path = [r["pos"]] + r["route"]
            if len(path) >= 2:
                xs = [p[0] for p in path]
                ys = [p[1] for p in path]
                self.route_lines[i].set_data(xs, ys)
            else:
                self.route_lines[i].set_data([], [])

        if done:
            plt.close(self.fig)

        return []

    def run(self):
        self.ani = FuncAnimation(
            self.fig, self._update,
            interval=self.interval_ms,
            blit=False,
            cache_frame_data=False
        )
        self._update(0)
        plt.show()
