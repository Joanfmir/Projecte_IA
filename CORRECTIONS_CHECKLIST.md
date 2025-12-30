# PR #12 - Checklist de Correcciones

## Estado Actual: ❌ RECHAZADO

---

## ✅ Correcciones YA Aplicadas (Durante Auditoría)

- [x] Eliminado spawn cutoff en `maybe_spawn_order()` que bloqueaba generación de pedidos
- [x] Añadidos comentarios explicativos en lógica de batching wait
- [x] Actualizado `after_fusion.json` con resultados post-fix
- [x] Generados reportes de auditoría (AUDIT_REPORT_PR12.md, INVESTIGATION_NOTES.md, EXECUTIVE_SUMMARY.md)

---

## ❌ Correcciones PENDIENTES (Para Aprobación)

### CRÍTICO #1: Validar Baseline Comparison
- [ ] Verificar en qué branch se ejecutó `baseline_pau.json`
  - ¿Fue en `pau_intent` o `rodolfo_intento`?
  - Si fue `pau_intent`, NO es comparable
  
- [ ] Ejecutar baseline válido en `rodolfo_intento` PRE-merge:
  ```bash
  git checkout rodolfo_intento  # ANTES de fusionar pau_intent
  python heuristic_benchmark.py \
    --output baseline_rodolfo_pre_merge.json \
    --seed 42 --episode_len 300 --width 25 --height 25 --riders 4 \
    --spawn 0.15 --max_eta 55 --block_size 5 --street_width 1 \
    --road_closure_prob 0.0 --road_closures_per_event 1 \
    --activation_cost 2.0 --batch_wait_ticks 0
  ```

- [ ] Documentar resultados de baseline_rodolfo_pre_merge:
  - Deliveries: ____
  - Reward: ____
  - Pending: ____
  - Distance: ____

### CRÍTICO #2: Investigar Degradación (Si Aplica)

**SI baseline_rodolfo_pre_merge da ~53 entregas:**
→ Hay REGRESIÓN real, investigar:

- [ ] Añadir logging detallado:
  ```python
  # En simulator.py
  print(f"Tick {self.t}: Orders generated={orders_count}, assigned={assigned_count}, pending={pending_count}")
  print(f"Tick {self.t}: Available riders={len(available_riders)}, waiting={waiting_count}")
  ```

- [ ] Ejecutar con logging y comparar PRE vs POST:
  - [ ] Tasa de generación de órdenes (órdenes/tick)
  - [ ] Tasa de asignación (assignments/tick)
  - [ ] Riders disponibles promedio
  - [ ] Tiempo de espera promedio en restaurante

- [ ] Identificar bottleneck específico:
  - [ ] ¿Se generan menos órdenes?
  - [ ] ¿Los riders están bloqueados/esperando demasiado?
  - [ ] ¿Las asignaciones son menos eficientes?
  - [ ] ¿Problema en get_available_riders()?

- [ ] Aplicar corrección específica según bottleneck

- [ ] Re-ejecutar benchmark y verificar mejora:
  ```bash
  python heuristic_benchmark.py --output after_fix.json [args]
  # Verificar: deliveries >= baseline_rodolfo_pre_merge
  ```

**SI baseline_rodolfo_pre_merge da ~36 entregas:**
→ NO hay regresión, rodolfo_intento ya tenía este performance
→ Documentar que pau_intent y rodolfo_intento son implementaciones diferentes
→ Explicar que la fusión mantiene el performance de rodolfo_intento

### CRÍTICO #3: Actualizar PR Description

- [ ] Añadir tabla comparativa:
  ```markdown
  ## Benchmark Results
  
  | Metric | baseline_rodolfo_pre | after_fusion | Change |
  |--------|---------------------|--------------|--------|
  | deliveries | __ | __ | __% |
  | ontime | __ | __ | __% |
  | late | __ | __ | __% |
  | pending | __ | __ | __ |
  | distance | __ | __ | __% |
  | reward | __ | __ | __% |
  
  **Analysis:** [Explicar cambios]
  ```

- [ ] Documentar comandos exactos:
  ```markdown
  ## Benchmark Commands
  
  ### Baseline (rodolfo_intento pre-merge):
  ```bash
  [comando exacto]
  ```
  
  ### After Fusion (rodolfo_intento post-merge):
  ```bash
  [comando exacto]
  ```
  ```

- [ ] Explicar cualquier diferencia o degradación observada

- [ ] Si hubo fixes post-merge, documentarlos

---

## ⚠️ Correcciones RECOMENDADAS (Opcional)

### Mejorar Robustez

- [ ] Añadir unit tests para batching wait:
  ```python
  def test_batching_wait_sets_wait_until():
      # Verificar que wait_until se setea correctamente
      
  def test_rider_waits_when_should():
      # Verificar que rider no se mueve durante wait
      
  def test_rider_can_receive_more_while_waiting():
      # Verificar batching funciona
  ```

- [ ] Simplificar lógica de batching si es demasiado compleja

- [ ] Revisar coordinación entre `get_available_riders()` y simulator

### Validación Adicional

- [ ] Ejecutar con múltiples seeds para validar consistencia:
  ```bash
  for seed in 42 43 44 45 46; do
    python heuristic_benchmark.py --output results_seed${seed}.json --seed $seed [otros args]
  done
  ```

- [ ] Calcular estadísticas (mediana, std dev) si hay varianza

---

## ✅ Criterios de Aprobación

El PR puede ser aprobado cuando:

1. ✅ Se ejecutó baseline válido en rodolfo_intento PRE-merge
2. ✅ Se comparó baseline vs after_fusion (mismo branch, solo diff la fusión)
3. ✅ Performance es IGUAL o MEJOR que baseline (o degradación explicada y aceptada)
4. ✅ PR description incluye tabla comparativa y análisis
5. ✅ Todos los tests pasan (si existen)
6. ✅ Código sigue estándares del proyecto

---

## 📝 Notas para Implementador

### Si Necesitas Ayuda

1. **Para entender causa raíz:** Lee INVESTIGATION_NOTES.md
2. **Para entender violaciones:** Lee AUDIT_REPORT_PR12.md
3. **Para quick summary:** Lee EXECUTIVE_SUMMARY.md
4. **Para seguir pasos:** Usa este checklist

### Comandos Útiles

```bash
# Ver diferencias entre branches
git diff pau_intent rodolfo_intento -- core/assignment_engine.py
git diff pau_intent rodolfo_intento -- core/dispatch_policy.py

# Ejecutar benchmark
python heuristic_benchmark.py --output test.json \
  --seed 42 --episode_len 300 --riders 4 --spawn 0.15 \
  --max_eta 55 --batch_wait_ticks 5 [otros args]

# Ver resultados
cat test.json | python -m json.tool

# Comparar JSONs
python -c "
import json
with open('baseline.json') as f: b = json.load(f)
with open('after.json') as f: a = json.load(f)
for k in b:
    if k in a:
        print(f'{k}: {b[k]} -> {a[k]} ({(a[k]-b[k])/b[k]*100:.1f}% change)' if isinstance(b[k], (int, float)) else f'{k}: {b[k]} -> {a[k]}')
"
```

### Próximos Pasos

1. Completa CRÍTICO #1 (baseline válido)
2. Basado en resultados, decide si necesitas CRÍTICO #2 (investigar degradación)
3. Completa CRÍTICO #3 (actualizar PR)
4. Re-solicita review cuando todos los ✅ estén marcados

---

**Última actualización:** 2025-12-30  
**Auditor:** GitHub Copilot Coding Agent

