# Projecte_IA

## Entrenamiento factorizado
- **Secuencial existente**: `python train_factored.py --episodes 500 --seed 7`
- **Paralelo actor-learner (nuevo)**: `python train_factored.py --parallel --n-workers 4 --episodes 500 --max-steps-per-episode 900 --sync-every 10 --log-every 10 --save-every 50 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay-steps 500`
  - Compatible con `artifacts/qtable_factored.pkl` y checkpoints previos (`--init-qpath` para arrancar desde uno distinto, `--qpath`/`--metrics-path` para rutas personalizadas).
  - El learner aplica los updates; los workers solo generan rollouts con snapshots inmutables.

### Flags clave
- `--n-workers`: nº de procesos de rollout (>=1).
- `--max-steps-per-episode`: pasos por episodio (por defecto `episode_len`).
- `--epsilon-start/--epsilon-end/--epsilon-decay-steps`: schedule lineal global por episodio.
- `--sync-every`: refresco de snapshot enviado a los workers.
- `--eval-every/--eval-episodes`: eval greedy (`epsilon=0`) con restauración automática.
- `--log-every`, `--save-every`, `--qpath`, `--metrics-path`: control de I/O.

### Schedule de epsilon
`epsilon(t) = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * min(t, decay_steps) / decay_steps)`, donde `t` es el episodio global (0-index). Monótono y acotado.

### Medición rápida (smoke, 4 episodios, 200 steps, seed=123)
| modo | workers | tiempo total | reward medio | wait_ratio | batching_eff | avg_rider_load |
| --- | --- | --- | --- | --- | --- | --- |
| Secuencial (`train_factored.py`) | 1 | 46.7s | -740.17 | 0.54 | 0.95 | 1.96 |
| Paralelo (`--parallel`) | 2 | 43.9s | -951.86 | 0.50 | 0.94 | 1.94 |

En ambos casos se mantuvo `epsilon_start=1.0` con schedule lineal hasta `epsilon_end`, seeds fijas y mismos parámetros de simulación. Ajusta episodios/steps para ejecuciones largas; la aceleración proviene del paralelismo y menor overhead de I/O.

## Heuristic benchmark (deterministic, apples-to-apples)
- Baseline (heuristic_mode=baseline, seed=42):  
  `python heuristic_benchmark.py --variant baseline --output baseline_rodolfo.json --seed 42 --episode_len 300 --width 25 --height 25 --riders 4 --spawn 0.15 --max_eta 55 --block_size 5 --street_width 1 --road_closure_prob 0.0 --road_closures_per_event 1 --activation_cost 2.0 --batch_wait_ticks 5`
- Fused heuristic (heuristic_mode=fusion, misma seed/config):  
  `python heuristic_benchmark.py --variant fusion --output after_fusion.json --seed 42 --episode_len 300 --width 25 --height 25 --riders 4 --spawn 0.15 --max_eta 55 --block_size 5 --street_width 1 --road_closure_prob 0.0 --road_closures_per_event 1 --activation_cost 2.0 --batch_wait_ticks 5`

Resultados (seed 42, 300 ticks):

| run              | reward_total | delivered_total | ontime | late | pending_end | distance_total |
| ---------------- | ------------ | ---------------- | ------ | ---- | ----------- | -------------- |
| baseline_rodolfo | -583.80      | 34               | 24     | 10   | 11          | 848.0          |
| after_fusion     | -744.76      | 36               | 22     | 14   | 1           | 835.0          |
