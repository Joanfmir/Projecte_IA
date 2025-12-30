from __future__ import annotations
import argparse
import json
import os
from dataclasses import fields

from simulation.simulator import Simulator, SimConfig
from core.dispatch_policy import A_ASSIGN_ANY_NEAREST, A_ASSIGN_URGENT_NEAREST


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, help="Ruta donde guardar el JSON de métricas")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=25)
    p.add_argument("--height", type=int, default=25)
    p.add_argument("--riders", type=int, default=4)
    p.add_argument("--episode_len", type=int, default=300)
    p.add_argument("--spawn", type=float, default=0.15)
    p.add_argument("--max_eta", type=int, default=55)
    p.add_argument("--block_size", type=int, default=5)
    p.add_argument("--street_width", type=int, default=1)
    p.add_argument("--road_closure_prob", type=float, default=0.0)
    p.add_argument("--road_closures_per_event", type=int, default=1)
    p.add_argument("--activation_cost", type=float, default=2.0)
    p.add_argument("--batch_wait_ticks", type=int, default=5)
    return p.parse_args()


def make_config(args) -> SimConfig:
    cfg_kwargs = dict(
        width=args.width,
        height=args.height,
        n_riders=args.riders,
        episode_len=args.episode_len,
        order_spawn_prob=args.spawn,
        max_eta=args.max_eta,
        seed=args.seed,
        block_size=args.block_size,
        street_width=args.street_width,
        road_closure_prob=args.road_closure_prob,
        road_closures_per_event=args.road_closures_per_event,
        activation_cost=args.activation_cost,
        batch_wait_ticks=args.batch_wait_ticks,
    )
    allowed = {f.name for f in fields(SimConfig)}
    filtered = {k: v for k, v in cfg_kwargs.items() if k in allowed}
    return SimConfig(**filtered)


def choose_action(sim: Simulator) -> int:
    pending = sim.om.get_pending_orders()
    urgent = [o for o in pending if getattr(o, "priority", 1) > 1 or o.is_urgent(sim.t)]
    return A_ASSIGN_URGENT_NEAREST if urgent else A_ASSIGN_ANY_NEAREST


def run_episode(sim: Simulator) -> dict:
    total_reward = 0.0
    done = False
    while not done:
        action = choose_action(sim)
        r, done = sim.step(action)
        total_reward += r

    snap = sim.snapshot()
    riders = sim.fm.get_all()
    distance_total = sum(getattr(r, "distance_travelled", 0.0) for r in riders)
    batch_wait = getattr(sim.cfg, "batch_wait_ticks", None)
    return {
        "seed": sim.cfg.seed,
        "config": {
            "width": sim.cfg.width,
            "height": sim.cfg.height,
            "n_riders": sim.cfg.n_riders,
            "episode_len": sim.cfg.episode_len,
            "order_spawn_prob": sim.cfg.order_spawn_prob,
            "max_eta": sim.cfg.max_eta,
            "block_size": sim.cfg.block_size,
            "street_width": sim.cfg.street_width,
            "road_closure_prob": sim.cfg.road_closure_prob,
            "road_closures_per_event": sim.cfg.road_closures_per_event,
            "activation_cost": sim.cfg.activation_cost,
            "batch_wait_ticks": batch_wait,
        },
        "reward_total": total_reward,
        "delivered_total": snap.get("delivered_total", 0),
        "delivered_ontime": snap.get("delivered_ontime", 0),
        "delivered_late": snap.get("delivered_late", 0),
        "pending_end": len(snap.get("pending_orders", [])),
        "ticks": snap.get("t", sim.t),
        "distance_total": distance_total,
    }


def main():
    args = parse_args()
    cfg = make_config(args)
    sim = Simulator(cfg)
    metrics = run_episode(sim)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
