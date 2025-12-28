# 🔍 Audit Report: Batching Strategy Implementation

**Fecha:** 2025-12-28  
**Auditor:** Arquitecto Senior / Lead QA  
**Rama auditada:** `copilot/audit-batching-strategy-implementation` (base: `rodolfo_intento`)

---

## ❌ RECHAZADO: Violaciones críticas detectadas

La implementación NO cumple con la especificación de "Batching Strategy" y presenta múltiples violaciones críticas que deben corregirse antes de aprobar el merge.

---

## Resumen de Violaciones

| # | Auditoría | Resultado | Severidad |
|---|-----------|-----------|-----------|
| 1 | Explosión Combinatoria | ⚠️ RIESGO MEDIO | Media |
| 2 | Invariantes Temporales | ❌ **FAIL** | **Crítica** |
| 3 | Coste Incremental/Inserción | ❌ **FAIL** | **Crítica** |
| 4 | Hardcodes y Capacidad | ❌ **FAIL** | **Crítica** |
| 5 | Aprendizaje en Espera | ✅ PASS | - |
| 6 | Consistencia Reward/Engine | ✅ PASS (N/A) | - |
| 7 | Determinismo | ⚠️ PARCIAL | Media |

---

## 1️⃣ Auditoría de Explosión Combinatoria (`core/factored_states.py`)

### Análisis de Bins por Feature

| Feature | Bins | Líneas |
|---------|------|--------|
| `bin_time` | 5 | L18-29 |
| `bin_pending_unassigned` | 5 | L32-42 |
| `bin_urgent` | 4 | L45-53 |
| `bin_free_riders` | 4 | L56-64 |
| `bin_min_slack` | 5 | L67-77 |
| `bin_zones_congested` | 4 | L80-88 |
| `bin_riders_at_restaurant` | 3 | L157-163 |
| `bin_min_rider_distance` | 4 | L166-174 |

**Producto Total Q1:** 5 × 5 × 4 × 4 × 5 × 4 × 3 × 4 = **96,000 estados**

### Resultado: ⚠️ RIESGO MEDIO

**Justificación:**
- El espacio de 96,000 estados es manejable dado que la Q-table es **sparse** (dict por estados visitados, líneas 44-45).
- NO se detecta precomputación densa ni arrays gigantes.
- La función `state_space_sizes()` (L433-445) es solo informativa.

**Observación (NO FAIL):**
- La especificación pedía `Empty/Partial/Full` para riders (3 categorías) y `closest_partial_eta_bin`. 
- El código actual usa `bin_free_riders` (4 bins) y `bin_min_rider_distance` pero **NO tiene la clasificación explícita Empty/Partial/Full** requerida.
- Esto es una **desviación de spec** pero no causa explosión combinatoria.

**Recomendación:**
Añadir features explícitos para contar riders por categoría de capacidad (Empty=0 pedidos, Partial=1-2, Full=3) según la especificación.

---

## 2️⃣ Auditoría de Invariantes Temporales (`simulation/simulator.py`)

### ❌ FAIL: Violación Crítica Detectada

**Archivo:** `simulation/simulator.py`  
**Función:** `apply_action()` (L581-634)  
**Líneas problemáticas:** L594-606, L611-623

```python
if action == A_ASSIGN_URGENT_NEAREST:
    # Bucle: asignar todos los urgentes posibles
    while True:  # ❌ VIOLACIÓN
        orders = self.om.get_pending_orders()
        riders = self.fm.get_all()
        pick = self.assigner.pick_urgent_nearest(orders, riders, now=self.t)
        if pick:
            o, r = pick
            self.assigner.assign(o, r)
            self._rebuild_plan_for_rider(r)
            assigned_count += 1
        else:
            break  # Sale cuando no hay más
    return assigned_count
```

**Error:** El método `apply_action()` contiene bucles `while True` que asignan **TODOS** los pedidos posibles en un solo tick, violando la especificación de "decisiones secuenciales por tick: ASSIGN (1 par pedido-rider) o WAIT".

**Por qué importa:**
- La especificación dice: "En cada tick, el agente toma **una sola decisión**"
- El bucle `while True` asigna múltiples pares (Order, Rider) en un único tick
- Esto elimina la oportunidad del agente de decidir si esperar para batching
- La física del simulador se rompe: múltiples asignaciones "gratis" en un tick

**Corrección requerida:**
```python
def apply_action(self, action: int) -> int:
    """
    Aplica la acción seleccionada.
    BATCHING CORRECTO: Asigna UN SOLO par (Pedido, Rider) por tick.
    """
    if action == A_ASSIGN_URGENT_NEAREST:
        orders = self.om.get_pending_orders()
        riders = self.fm.get_all()
        pick = self.assigner.pick_urgent_nearest(orders, riders, now=self.t)
        if pick:
            o, r = pick
            self.assigner.assign(o, r)
            self._rebuild_plan_for_rider(r)
            return 1
        return 0

    if action == A_ASSIGN_ANY_NEAREST:
        orders = self.om.get_pending_orders()
        riders = self.fm.get_all()
        pick = self.assigner.pick_any_nearest(orders, riders)
        if pick:
            o, r = pick
            self.assigner.assign(o, r)
            self._rebuild_plan_for_rider(r)
            return 1
        return 0
    
    # ... resto igual
```

### Verificación adicional - `self.t += 1`:
✅ **CORRECTO:** El incremento de tiempo (`self.t += 1`) ocurre exactamente una vez por llamada a `step()` (L665), tanto para ASSIGN como para WAIT.

---

## 3️⃣ Auditoría de Coste Incremental/Inserción (`core/assignment_engine.py`)

### ❌ FAIL: No calcula Δcost

**Archivo:** `core/assignment_engine.py`  
**Funciones:** `pick_any_nearest()` (L76-98), `pick_urgent_nearest()` (L100-129)

```python
def pick_any_nearest(self, orders: List[Order], riders: List[Rider]) -> ...:
    # ...
    for o in orders:
        for r in riders:
            if r.position == self.restaurant_pos:
                eta = self._eta_octile_restaurant_to_drop(o)  # ❌
            else:
                eta = self._eta_octile_rider_to_drop(r, o)    # ❌
            if eta < best_eta:
                best_eta = eta
                best = (o, r)
```

**Error:** La selección de candidatos usa **distancia absoluta** (ETA rider → pedido) en vez de **Δcost** (costo de inserción en ruta existente).

**Por qué importa:**
- Para riders `Partial` (con 1-2 pedidos ya asignados), la métrica correcta es:
  - `Δcost = cost(ruta_con_nuevo_pedido) - cost(ruta_actual)`
- El código actual calcula `Distancia(Rider → restaurant → dropoff)` sin considerar:
  - La ruta existente del rider
  - El desvío que causaría insertar el nuevo pedido
  - El impacto en los pedidos ya asignados

**Ejemplo del problema:**
- Rider A tiene pedido para dropoff (10, 5) y está en restaurante
- Nuevo pedido llega con dropoff (10, 6) (muy cerca del primero)
- Otro rider B está libre pero lejos
- El código actual puede elegir B (menor ETA absoluta) cuando A es mejor opción (menor Δcost)

**Corrección requerida:**
```python
def _calculate_insertion_delta(self, rider: Rider, order: Order) -> float:
    """
    Calcula el delta de costo al insertar un pedido en la ruta del rider.
    Para riders Empty: costo = ETA absoluta
    Para riders Partial: costo = cost(ruta_nueva) - cost(ruta_actual)
    """
    current_orders = rider.assigned_order_ids
    
    if not current_orders:  # Rider vacío
        return self._eta_octile_rider_to_drop(rider, order)
    
    # Calcular costo actual de la ruta
    current_cost = self._calculate_route_cost(rider)
    
    # Calcular costo con el nuevo pedido insertado (en mejor posición)
    new_cost = self._calculate_route_cost_with_insertion(rider, order)
    
    return new_cost - current_cost

def pick_any_nearest(self, orders: List[Order], riders: List[Rider]) -> ...:
    # ...
    for o in orders:
        for r in riders:
            delta = self._calculate_insertion_delta(r, o)  # ✅ Delta
            if delta < best_delta:
                best_delta = delta
                best = (o, r)
```

### Verificación de precedencia pickup→dropoff:
El código SÍ respeta la precedencia pickup→dropoff en `_rebuild_plan_for_rider()` (L236-256):
- Primero va al restaurante si no ha recogido
- Luego hace entregas en orden EDF (Earliest Deadline First)
- Finalmente vuelve al restaurante

---

## 4️⃣ Auditoría de Hardcodes y Capacidad (`core/fleet_manager.py`)

### ❌ FAIL: Múltiples violaciones

**Violación 1: Capacidad no es 3**

**Archivo:** `core/fleet_manager.py`  
**Línea:** 22

```python
capacity: int = 2  # ❌ Debería ser 3 según spec
```

**Error:** La especificación indica "capacidad=3 por rider" pero el código usa `capacity=2`.

---

**Violación 2: Hardcode en factored_states.py**

**Archivo:** `core/factored_states.py`  
**Línea:** 235

```python
def is_eligible(r):
    has_capacity = len(r.get("assigned", [])) < 2  # ❌ Hardcode literal
    # ...
```

**Error:** Condición literal `< 2` en vez de usar `rider.capacity` o equivalente.

**Por qué importa:**
- Si cambio `capacity=3` en fleet_manager.py pero olvido actualizar esta línea, el sistema se rompe
- El conteo de "elegibles" en la codificación de estados no coincidirá con la realidad
- Violación del principio DRY (Don't Repeat Yourself)

**Corrección requerida:**

1. En `core/fleet_manager.py` L22:
```python
capacity: int = 3  # ✅ Según spec
```

2. En `core/factored_states.py` L234-235:
```python
def is_eligible(r):
    # Obtener capacidad de la configuración centralizada
    RIDER_CAPACITY = 3  # O mejor: importar desde un config
    has_capacity = len(r.get("assigned", [])) < RIDER_CAPACITY
    # ...
```

**Mejor aún:** Añadir la capacidad al snapshot del rider y usarla dinámicamente:
```python
has_capacity = len(r.get("assigned", [])) < r.get("capacity", 3)
```

---

## 5️⃣ Auditoría de Aprendizaje en Espera (`core/factored_q_agent.py`)

### ✅ PASS

**Verificación 1: No hay early return para WAIT**

El método `update()` (L192-218) no hace early return cuando `last_action == WAIT`:

```python
def update(self, snap: Dict, action: int, reward: float, snap_next: Dict, done: bool) -> None:
    if self.last_q_used == "none":  # ✅ Solo salta si NO se usó tabla
        return
    
    # ... actualiza Q normalmente para cualquier acción incluyendo WAIT
```

**Verificación 2: WAIT está en acciones válidas**

```python
def _valid_actions_q1(self, features: Dict) -> List[int]:
    # ...
    valid.append(A_WAIT)  # ✅ Siempre válido como fallback
    return valid
```

**Verificación 3: La transición de WAIT actualiza Q**

Cuando el agente elige WAIT, `last_q_used = "Q1"` (L141) y la actualización procede normalmente en `update()` con el reward del tick (incluyendo penalizaciones por pedidos pendientes de L568-569 en simulator.py).

---

## 6️⃣ Auditoría de Consistencia Reward vs Engine

### ✅ PASS (N/A)

**Observación:** No existe archivo `core/reward.py` ni parámetro `activation_cost` en el código.

El reward se calcula en `Simulator.compute_reward()` (L540-576) y no hay concepto de "activation penalty" para encender riders nuevos.

**Implicación:** Esta auditoría no aplica al estado actual del código. Si se añade activation_cost en el futuro, debe centralizarse.

---

## 7️⃣ Auditoría de Determinismo

### ⚠️ PARCIAL

**✅ Seeds fijas en tests:**
- `test_sanity_check.py` L35: `seed=42`
- `train_factored.py` L40: `seed=base_seed`

**⚠️ Tie-breakers no deterministas:**

**Archivo:** `core/factored_q_agent.py`  
**Línea:** 80

```python
def best_action(self, q_table: Dict, state: Tuple, actions: List[int]) -> int:
    q_values = [(a, self.get_q(q_table, state, a)) for a in actions]
    max_q = max(v for _, v in q_values)
    best_actions = [a for a, v in q_values if v == max_q]
    return self.rng.choice(best_actions)  # ⚠️ Aleatorio en empates
```

**Problema:** El tie-breaker es aleatorio (`rng.choice`). Aunque el RNG tiene seed fija, esto puede causar comportamiento diferente si el orden de `actions` cambia.

**Corrección recomendada:**
```python
def best_action(self, q_table: Dict, state: Tuple, actions: List[int]) -> int:
    q_values = [(a, self.get_q(q_table, state, a)) for a in actions]
    max_q = max(v for _, v in q_values)
    best_actions = sorted([a for a, v in q_values if v == max_q])  # ✅ Ordenar
    return best_actions[0]  # ✅ Determinista: menor acción
```

---

**Archivo:** `core/assignment_engine.py`  
**Funciones:** `pick_any_nearest()`, `pick_urgent_nearest()`

```python
for o in orders:
    for r in riders:
        if eta < best_eta:  # ⚠️ Sin tie-breaker por ID
            best_eta = eta
            best = (o, r)
```

**Problema:** Cuando hay empate en ETA, la selección depende del orden de iteración de las listas.

**Corrección requerida:**
```python
for o in orders:
    for r in riders:
        if eta < best_eta or (eta == best_eta and (o.order_id, r.rider_id) < (best[0].order_id, best[1].rider_id)):
            best_eta = eta
            best = (o, r)
```

---

## Resumen de Correcciones Requeridas

### Prioridad CRÍTICA (bloquean merge):

1. **`simulation/simulator.py::apply_action()`** - Eliminar bucles `while True`, asignar solo 1 par por tick.

2. **`core/assignment_engine.py::pick_*_nearest()`** - Implementar cálculo de Δcost para inserción en riders Partial.

3. **`core/fleet_manager.py::Rider`** - Cambiar `capacity: int = 2` a `capacity: int = 3`.

4. **`core/factored_states.py::extract_features()`** - Reemplazar hardcode `< 2` con referencia a capacidad centralizada.

### Prioridad MEDIA (recomendadas):

5. **`core/factored_q_agent.py::best_action()`** - Usar tie-breaker determinista.

6. **`core/assignment_engine.py`** - Añadir tie-breaker por ID en selección.

7. **`core/factored_states.py`** - Añadir features Empty/Partial/Full según spec original.

---

## Conclusión

**❌ RECHAZADO**

La implementación presenta 4 violaciones críticas que rompen invariantes fundamentales:
- La física del simulador (múltiples asignaciones por tick)
- La lógica de batching (sin cálculo de delta de inserción)
- La configuración de capacidad (2 vs 3)
- El acoplamiento hardcodeado

Se recomienda rechazar este PR hasta que se implementen las correcciones marcadas como CRÍTICAS.

---

*Generado por Arquitecto Senior de Software / Lead QA*
