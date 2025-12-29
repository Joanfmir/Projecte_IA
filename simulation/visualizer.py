# simulation/visualizer.py
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from core.dispatch_policy import A_ASSIGN_ANY_NEAREST


def ticks_to_time(t: int, episode_len: int) -> str:
    """
    Convierte ticks de simulación a formato hora HH:MM.
    Empieza a las 19:00 y termina a las 00:00 (5 horas = 300 minutos).
    """
    total_minutes = 300  # 5 horas (19:00 a 00:00)
    minutes_elapsed = (t / episode_len) * total_minutes
    
    start_hour = 19
    start_minutes = start_hour * 60  # 19:00 = 1140 minutos desde medianoche
    
    current_minutes = start_minutes + minutes_elapsed
    
    # Si pasa de medianoche (1440 minutos)
    if current_minutes >= 1440:
        current_minutes -= 1440
    
    hours = int(current_minutes // 60)
    mins = int(current_minutes % 60)
    
    return f"{hours:02d}:{mins:02d}"


class Visualizer:
    """
    Visualizer optimizado + robusto:
    - No recrea objetos por frame (evita ralentización)
    - Stats se actualizan cada N ticks
    - Rutas se recortan a K pasos
    - Tráfico por zonas: soporta varias keys y fallback a tráfico global
    """

    def __init__(
        self,
        sim,
        policy=None,
        interval_ms: int = 220,
        stats_every: int = 5,
        route_draw_limit: int = 40,
    ):
        self.sim = sim
        self.policy = policy
        self.interval_ms = interval_ms
        self.stats_every = max(1, int(stats_every))
        self.route_draw_limit = max(5, int(route_draw_limit))
        self.episode_len = sim.cfg.episode_len

        snap = self.sim.snapshot()
        self.W = snap["width"]
        self.H = snap["height"]
        self.buildings = set(tuple(x) for x in snap.get("buildings", []))
        self.avenues = snap.get("avenues", [])

        # FIG / AX principal
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(top=0.88, right=0.78)
        
        # Pantalla completa
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')  # Windows
        except:
            try:
                mng.full_screen_toggle()  # Alternativa
            except:
                pass

        self.ax.set_xlim(-0.5, self.W - 0.5)
        self.ax.set_ylim(-0.5, self.H - 0.5)
        self.ax.set_aspect("equal")

        # Fondo + grid
        self.ax.set_facecolor("#f5f5f5")
        self.ax.set_xticks(range(self.W))
        self.ax.set_yticks(range(self.H))
        self.ax.grid(True, linewidth=1.0, alpha=0.10)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.tick_params(length=0)

        # Estático
        self._draw_buildings()
        self._draw_avenues()

        # Cierres (rojo)
        self.closed_lc = LineCollection([], linewidths=3.0, alpha=0.9, colors="#d62728")
        self.ax.add_collection(self.closed_lc)

        self.ax.set_title("Pizza Delivery IA", fontsize=18, pad=8)

        # HUD
        self.hud = self.fig.text(
            0.02, 0.96, "",
            fontsize=11, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.75, edgecolor="none")
        )

        # Capas dinámicas
        self.shop = self.ax.scatter([], [], s=320, marker="s", color="#ff7f0e")
        self.orders_normal = self.ax.scatter([], [], s=90, marker="o", color="#2ca02c")
        self.orders_urgent = self.ax.scatter([], [], s=160, marker="o", color="#d62728")

        self.rider_scatters = []
        self.route_lines = []
        self.rider_labels = []

        # Leyendas (1 vez)
        self.legend_main = self._build_main_legend()
        self.ax.add_artist(self.legend_main)

        self.zone_patches, self.zone_legend, self.zone_texts = self._build_zone_legend()
        self.ax.add_artist(self.zone_legend)
        self._draw_zone_labels()

        # Panel stats a la izquierda
        self.ax_stats = self.fig.add_axes([0.02, 0.06, 0.23, 0.40])
        self.ax_stats.axis("off")
        self.ax_stats.set_title("Stats (tiempo real)", fontsize=12)
        self.stats_text = self.ax_stats.text(
            0.0, 1.0, "",
            va="top", ha="left",
            fontsize=9,
            family="monospace"
        )

        # Panel "App del Rider" a la derecha (más arriba)
        self.ax_rider_app = self.fig.add_axes([0.82, 0.15, 0.17, 0.45])
        self.ax_rider_app.axis("off")
        self.ax_rider_app.set_title("App Riders", fontsize=11, fontweight="bold")
        self.rider_app_text = self.ax_rider_app.text(
            0.0, 1.0, "",
            va="top", ha="left",
            fontsize=8,
            family="monospace"
        )

        # init
        self._update_zone_legend(snap)
        self._update_stats_panel(snap)
        self._update_rider_app(snap)
        self._last_stats_t = -10**9

    # -------------------------
    # Estático
    # -------------------------
    def _draw_buildings(self):
        for (x, y) in self.buildings:
            rect = Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=0,
                             facecolor="#273043", alpha=0.35)
            self.ax.add_patch(rect)

    def _draw_road_closures(self, blocked_nodes):
        """Dibuja los cierres de calle como cuadrados naranjas (obras)."""
        # Limpiar cierres anteriores
        if hasattr(self, '_closure_patches'):
            for p in self._closure_patches:
                p.remove()
        self._closure_patches = []
        
        for node in blocked_nodes:
            if isinstance(node, (list, tuple)) and len(node) == 2:
                x, y = node
                rect = Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=1,
                                 facecolor="#e74c3c", edgecolor="#c0392b", alpha=0.7)
                self.ax.add_patch(rect)
                self._closure_patches.append(rect)

    def _draw_avenues(self):
        W, H = self.W, self.H
        for av in self.avenues:
            m = av.get("m", 1.0)
            b = av.get("b", 0.0)

            xs, ys = [], []
            for x in range(W):
                y = m * x + b
                if 0 <= y <= H - 1:
                    xs.append(x)
                    ys.append(y)

            if len(xs) >= 2:
                self.ax.plot(xs, ys, linewidth=3.5, alpha=0.10, color="#666666")

    def _build_main_legend(self):
        handles = [
            Line2D([0], [0], marker="s", linestyle="None", markersize=11,
                   markerfacecolor="#ff7f0e", markeredgecolor="none", label="Restaurante"),
            Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                   markerfacecolor="#2ca02c", markeredgecolor="none", label="Pedido (normal)"),
            Line2D([0], [0], marker="o", linestyle="None", markersize=10,
                   markerfacecolor="#d62728", markeredgecolor="none", label="Pedido (urgente)"),
            Line2D([0], [0], marker="D", linestyle="None", markersize=10,
                   markerfacecolor="#1f77b4", markeredgecolor="white", label="Repartidor"),
            Line2D([0], [0], linewidth=2, color="#1f77b4", linestyle="--", label="Ruta (entregando)"),
            Line2D([0], [0], linewidth=2, color="#e74c3c", linestyle="--", label="Ruta (volviendo)"),
            Line2D([0], [0], linewidth=3, color="#d62728", label="Calle cortada"),
            Rectangle((0, 0), 1, 1, facecolor="#273043", alpha=0.35, label="Edificio / manzana"),
        ]
        return self.ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            title="Leyenda",
            borderaxespad=0.0
        )

    def _build_zone_legend(self):
        patches = [
            Patch(facecolor="#cfe8ff", edgecolor="none", label="Z0 (TL): ?"),
            Patch(facecolor="#d7ffd7", edgecolor="none", label="Z1 (TR): ?"),
            Patch(facecolor="#fff2cc", edgecolor="none", label="Z2 (BL): ?"),
            Patch(facecolor="#ffd6d6", edgecolor="none", label="Z3 (BR): ?"),
        ]
        leg = self.ax.legend(
            handles=patches,
            loc="upper left",
            bbox_to_anchor=(1.01, 0.55),
            frameon=True,
            title="Tráfico por zonas",
            borderaxespad=0.0
        )
        return patches, leg, leg.get_texts()

    def _draw_zone_labels(self):
        W, H = self.W, self.H
        fs = 14
        self.ax.text(W * 0.25, H * 0.75, "Z0 (TL)", fontsize=fs, weight="bold",
                     ha="center", va="center", alpha=0.25)
        self.ax.text(W * 0.75, H * 0.75, "Z1 (TR)", fontsize=fs, weight="bold",
                     ha="center", va="center", alpha=0.25)
        self.ax.text(W * 0.25, H * 0.25, "Z2 (BL)", fontsize=fs, weight="bold",
                     ha="center", va="center", alpha=0.25)
        self.ax.text(W * 0.75, H * 0.25, "Z3 (BR)", fontsize=fs, weight="bold",
                     ha="center", va="center", alpha=0.25)

    # -------------------------
    # Tráfico por zonas (ROBUSTO)
    # -------------------------
    def _extract_zone_levels(self, snap: dict) -> dict:
        """
        Devuelve dict {0: 'low', 1:'high', 2:'medium', 3:'low'}.
        Soporta varias keys y fallback al tráfico global.
        """
        zones = None

        # posibles nombres que puede traer tu snapshot
        for k in ("traffic_zones", "zone_traffic", "zone_levels", "zones_traffic"):
            v = snap.get(k, None)
            if isinstance(v, dict) and v:
                zones = v
                break

        # si viene como lista/tupla de 4 strings
        if zones is None:
            v = snap.get("traffic_zones", None)
            if isinstance(v, (list, tuple)) and len(v) == 4:
                zones = {i: v[i] for i in range(4)}

        # fallback: usar tráfico global en todas las zonas
        if zones is None or not isinstance(zones, dict):
            g = snap.get("traffic", snap.get("traffic_level", "mixed"))
            zones = {0: g, 1: g, 2: g, 3: g}

        # normalizamos keys por si vienen como "0","1"
        out = {}
        for i in range(4):
            if i in zones:
                out[i] = zones[i]
            elif str(i) in zones:
                out[i] = zones[str(i)]
            else:
                out[i] = snap.get("traffic", snap.get("traffic_level", "mixed"))
        return out

    def _update_zone_legend(self, snap: dict):
        zones = self._extract_zone_levels(snap)

        def name(z: int) -> str:
            return "TL" if z == 0 else "TR" if z == 1 else "BL" if z == 2 else "BR"

        labels = [f"Z{z} ({name(z)}): {zones.get(z, '?')}" for z in (0, 1, 2, 3)]
        for i in range(min(len(self.zone_texts), len(labels))):
            self.zone_texts[i].set_text(labels[i])

    # -------------------------
    # Stats (barato)
    # -------------------------
    def _update_stats_panel(self, snap: dict):
        t = snap["t"]
        riders = snap.get("riders", [])
        orders_full = snap.get("orders_full", [])

        time_str = ticks_to_time(t, self.episode_len)
        lines = []
        lines.append(f"Hora: {time_str}")
        lines.append("")
        lines.append("RIDERS")
        lines.append("------")

        for r in riders:
            name = r.get("name", f"R{r['id']}")
            x, y = r["pos"]

            avail = r.get("available", True)
            route_len = len(r.get("route", []))

            carrying = r.get("carrying", None)
            if carrying is None:
                carry_txt = "-"
            elif isinstance(carrying, list):
                carry_txt = ",".join(map(str, carrying))
            else:
                carry_txt = str(carrying)

            lines.append(
                f"{name:6s} pos=({x:2d},{y:2d}) av={int(avail)} carry={carry_txt:<5} route={route_len:3d}"
            )

        lines.append("")
        lines.append("ORDERS (últimos 8)")
        lines.append("-----------------")

        def sort_key(o):
            st = o.get("status", "pending")
            pr = o.get("priority", 1)
            st_rank = 0 if st in ("pending", "picked") else 1
            created = o.get("created_at", 0) if o.get("created_at", None) is not None else 0
            return (st_rank, -pr, -created)

        top = sorted(orders_full, key=sort_key)[:8]
        for o in top:
            oid = o["id"]
            pr = o.get("priority", 1)
            st = o.get("status", "pending")
            assigned = o.get("assigned_to", None)
            created = o.get("created_at", None)
            deadline = o.get("deadline", None)
            delivered = o.get("delivered_at", None)

            asg_txt = "-" if assigned is None else str(assigned)
            since = "-" if created is None else str(t - created)
            rem = "-" if deadline is None else str(deadline - t)
            total = "-" if (created is None or delivered is None) else str(delivered - created)
            delay = "-" if (delivered is None or deadline is None) else str(delivered - deadline)

            lines.append(f"O{oid:02d} P{pr} st={st:<8} asg={asg_txt:<2} since={since:>3} rem={rem:>4} total={total:>3} dly={delay:>4}")

        self.stats_text.set_text("\n".join(lines))

    def _update_rider_app(self, snap: dict):
        """Muestra la 'app' de cada rider con sus pedidos asignados."""
        t = snap["t"]
        riders = snap.get("riders", [])
        orders_full = snap.get("orders_full", [])
        restaurant = snap.get("restaurant", (0, 0))
        
        # Crear diccionario de pedidos por ID para búsqueda rápida
        orders_by_id = {o["id"]: o for o in orders_full}
        
        lines = []
        for r in riders:
            name = r.get("name", f"R{r['id']}")
            pos = tuple(r["pos"])
            carrying = r.get("carrying", None)  # ID del pedido que lleva actualmente
            assigned_ids = r.get("assigned", [])  # Campo correcto del snapshot
            resting = r.get("resting", False)
            picked = r.get("picked", False)  # Si ya recogió los pedidos
            
            lines.append(f"+------------------+")
            lines.append(f"|  {name:^14s}  |")
            lines.append(f"+------------------+")
            
            if resting:
                lines.append(f"| [ZZZ] Descansando |")
                lines.append(f"+------------------+")
                lines.append("")
                continue
            
            if not assigned_ids:
                lines.append(f"| [OK] Sin pedidos  |")
                lines.append(f"|    Esperando...   |")
                lines.append(f"+------------------+")
                lines.append("")
                continue
            
            # Mostrar hacia dónde va
            if picked and not assigned_ids:
                lines.append(f"| >> Volviendo      |")
            elif not picked:
                lines.append(f"| >> Recogiendo...  |")
            else:
                lines.append(f"| >> Entregando     |")
            
            for oid in assigned_ids:
                o = orders_by_id.get(oid)
                if o is None:
                    continue
                    
                dropoff = o.get("dropoff", (0, 0))
                priority = o.get("priority", 1)
                deadline = o.get("deadline", 0)
                status = o.get("status", "pending")
                
                tiempo_rest = deadline - t
                prio_txt = "[!]" if priority > 1 else "   "
                
                # Indicar si es el pedido actual
                current = ">>" if carrying == oid else "  "
                
                lines.append(f"|{current}{prio_txt} Pedido {oid:<3}   |")
                lines.append(f"|    Dest: ({dropoff[0]:2},{dropoff[1]:2})   |")
                if tiempo_rest > 0:
                    lines.append(f"|    Tiempo: {tiempo_rest:3}t    |")
                else:
                    lines.append(f"|    !! TARDE !!    |")
                lines.append(f"+------------------+")
            
            lines.append("")
        
        self.rider_app_text.set_text("\n".join(lines))

    # -------------------------
    # Dinámica
    # -------------------------
    def _ensure_riders(self, n: int):
        while len(self.rider_scatters) < n:
            # Marcador de diamante para representar la moto
            sc = self.ax.scatter([], [], s=120, marker="D", color="#1f77b4", edgecolors="white", linewidths=1.2, zorder=10)
            self.rider_scatters.append(sc)
            (ln,) = self.ax.plot([], [], linewidth=2, color="#1f77b4", linestyle="--")
            self.route_lines.append(ln)
            txt = self.ax.text(0, 0, "", fontsize=8, ha="left", va="bottom", fontweight="bold")
            self.rider_labels.append(txt)

    def _safe_offsets(self, pts):
        return pts if pts else [[-1000, -1000]]

    def _update(self, _frame):
        # acción
        if self.policy is not None:
            snap0 = self.sim.snapshot()
            if hasattr(self.policy, "choose_action_snapshot"):
                a = self.policy.choose_action_snapshot(snap0)
            else:
                s = self.sim.compute_state()
                a = self.policy.choose_action(s)
        else:
            a = A_ASSIGN_ANY_NEAREST

        _, done = self.sim.step(a)

        snap = self.sim.snapshot()
        t = snap["t"]

        orders = snap.get("pending_orders", [])
        riders = snap.get("riders", [])
        traffic = snap.get("traffic", "mixed")
        closures = snap.get("closures", 0)
        blocked = snap.get("blocked", 0)

        time_str = ticks_to_time(t, self.episode_len)
        self.hud.set_text(
            f"🕐 {time_str}   pending={len(orders)}   riders={len(riders)}   traffic={traffic}   closures={closures}   blocked={blocked}"
        )

        # restaurante
        sx, sy = snap["restaurant"]
        self.shop.set_offsets([[sx, sy]])

        # pedidos
        normal_pts, urgent_pts = [], []
        for (loc, priority, *_rest) in orders:
            (urgent_pts if priority > 1 else normal_pts).append([loc[0], loc[1]])
        self.orders_normal.set_offsets(self._safe_offsets(normal_pts))
        self.orders_urgent.set_offsets(self._safe_offsets(urgent_pts))

        # cierres dinámicos (como cuadrados naranjas = obras)
        blocked_nodes = snap.get("blocked_nodes", [])
        self._draw_road_closures(blocked_nodes)

        # zonas
        self._update_zone_legend(snap)

        # riders + rutas
        self._ensure_riders(len(riders))
        for i, r in enumerate(riders):
            x, y = r["pos"]
            # Actualizar posición del rider
            self.rider_scatters[i].set_offsets([[x, y]])

            name = r.get("name", f"R{r['id']}")
            carrying = r.get("carrying", None)
            if carrying is None:
                lab = name
            elif isinstance(carrying, list):
                lab = f"{name}>{','.join(map(str, carrying))}"
            else:
                lab = f"{name}>{carrying}"

            self.rider_labels[i].set_position((x + 0.25, y + 0.25))
            self.rider_labels[i].set_text(lab)

            route = r.get("route", [])[: self.route_draw_limit]
            path = [r["pos"]] + route
            
            # Determinar color: azul si va a entregar, rojo si vuelve al restaurante
            picked = r.get("picked", False)
            has_orders = len(r.get("assigned", [])) > 0
            
            if picked and not has_orders:
                # Volviendo al restaurante (ya entregó todo)
                route_color = "#e74c3c"  # Rojo
            elif not picked and has_orders:
                # Yendo a recoger
                route_color = "#1f77b4"  # Azul
            elif picked and has_orders:
                # Entregando pedidos
                route_color = "#1f77b4"  # Azul
            else:
                # Sin pedidos, disponible
                route_color = "#1f77b4"  # Azul
            
            self.route_lines[i].set_color(route_color)
            
            if len(path) >= 2:
                xs = [p[0] for p in path]
                ys = [p[1] for p in path]
                self.route_lines[i].set_data(xs, ys)
            else:
                self.route_lines[i].set_data([], [])

        # stats panel cada N ticks
        if (t - self._last_stats_t) >= self.stats_every:
            self._update_stats_panel(snap)
            self._last_stats_t = t

        # App del rider se actualiza cada frame para que sea responsive
        self._update_rider_app(snap)

        if done:
            plt.close(self.fig)

        return []

    def run(self):
        self.ani = FuncAnimation(
            self.fig,
            self._update,
            interval=self.interval_ms,
            blit=False,
            cache_frame_data=False
        )
        self._update(0)
        plt.show()
