# Pizza Delivery RL

Este proyecto implementa un entorno de simulación para el problema de **Delivery de Pizzas** y una solución basada en **Reinforcement Learning Factorizado (Factored Q-Learning)**.

El agente gestiona una flota de repartidores para recoger pedidos en un restaurante central y entregarlos a clientes distribuidos en una grilla urbana, minimizando tiempos de entrega y gestionando eventos dinámicos como tráfico y cortes de calle.

## Inicio Rápido

### Requisitos
- Python 3.8+
- Librerías: `numpy`, `matplotlib`, `pandas`, `tqdm`.

### Estructura del Proyecto
- `main_factored.py`: Script principal para visualización e inferencia.
- `train_factored.py`: Script de entrenamiento paralelo/secuencial.
- `eval_factored.py`: Script de evaluación reproducible y métricas detalladas.
- `core/`: Lógica del agente (Q-Learning factorizado, encoding de estados, grafo vial).
- `simulation/`: Motor de simulación y visualizador.

---

## Entrenamiento del Agente

Se puede entrenar al agente desde cero en modo secuencial o paralelo (multiprocessing).

```bash
# Entrenamiento estándar (secuencial)
python train_factored.py --episodes 1000 --save-every 100

# Entrenamiento paralelo (más rápido)
python train_factored.py --parallel --n-workers 4 --episodes 2000
```

**Argumentos clave:**
- `--out-dir`: Carpeta para guardar checkpoints y métricas (default: `artifacts/`).
- `--episode_len`: Duración de cada episodio en ticks (default: 900).
- `--epsilon-start` / `--epsilon-end`: Control de exploración.

El entrenamiento generará dos archivos importantes en `artifacts/`:
1. `qtable_factored.pkl`: La tabla Q aprendida.
2. `metrics_factored.csv`: Histórico de recompensas y estadísticas.

---

## Evaluación y Métricas

Para evaluar el rendimiento del agente entrenado frente a una baseline heurística (asignación al más cercano):

```bash
python eval_factored.py --n-episodes 50 --qpath artifacts/qtable_factored.pkl
```

Este script genera un reporte en consola comparando:
- Tasa de entregas a tiempo (On-time ratio).
- Recompensa media.
- Fatiga promedio de los riders.

---

## Visualización y Demo

Para ver al agente en acción con una interfaz gráfica (Matplotlib):

```bash
python main_factored.py --visual --policy trained --qpath artifacts/qtable_factored.pkl
```

**Controles:**
- La ventana muestra el mapa, los riders, pedidos urgentes (círculos rojos) y normales (verdes).
- A la derecha se ve una "App" simulada con el estado de cada rider.
- **Leyenda:**
  - **Línea azul:** Ruta de entrega.
  - **Línea roja:** Ruta de retorno a base.
  - **Cuadrados naranjas:** Calles cortadas (obras).

---

## Detalles Técnicos

### Enfoque Factorizado
El agente descompone el problema en sub-decisiones para manejar la complejidad:
1. **Asignación (Q1):** ¿A qué rider asignar un nuevo pedido? (Minimizar costes individuales).
2. **Re-planificación (Q3):** ¿Debe un rider cambiar su ruta debido al tráfico?

El estado se codifica (`FactoredStateEncoder`) discretizando variables como: tiempo restante, ubicación, carga actual y tráfico por zonas.

### Simulación
- **Grilla:** Mapa de Manhattan con edificios y sentidos de calles.
- **Tráfico:** Dinámico, afecta la velocidad de movimiento en los 4 cuadrantes.
- **Eventos:** Aparición estocástica de pedidos y cortes de calle temporales.
