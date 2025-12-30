# Investigation Notes: PR #12 Performance Degradation

## Problem Summary
After merging heuristic from `pau_intent` to `rodolfo_intento`, deliveries dropped 32% (53 → 36).

## Testing Performed

### Test 1: Original after_fusion (with spawn cutoff bug)
```bash
python heuristic_benchmark.py --output after_fusion.json \
  --seed 42 --episode_len 300 --width 25 --height 25 --riders 4 \
  --spawn 0.15 --max_eta 55 --batch_wait_ticks 5
```
**Results:** 36 deliveries, 1 pending, reward -744.76

### Test 2: After fixing spawn cutoff
```bash
# Same command
```
**Results:** 36 deliveries, 9 pending, reward -837.76
**Analysis:** More orders generated (pending went up) but still not delivered

### Test 3: No batching wait (batch_wait_ticks=0)
```bash
python heuristic_benchmark.py --output /tmp/test_no_batching.json \
  --seed 42 --episode_len 300 --batch_wait_ticks 0 [other args same]
```
**Results:** 34 deliveries, 11 pending, reward -583.80
**Analysis:** Even WITHOUT batching wait, deliveries are still low (34 vs 53)

### Test 4: Different seed (43)
```bash
python heuristic_benchmark.py --seed 43 [other args same]
```
**Results:** 36 deliveries, 8 pending
**Analysis:** Consistently low deliveries across seeds

## Key Metrics Comparison

| Metric | baseline_pau | after_fusion | after (no batch) | Change |
|--------|--------------|--------------|------------------|--------|
| Deliveries | 53 | 36 | 34 | -32% to -36% |
| Pending | 0 | 9 | 11 | ∞ |
| Distance | 964 | 835 | 848 | -13% to -12% |
| Dist/delivery | 18.2 | 23.2 | 24.9 | +27% to +37% |
| Reward | -2503 | -838 | -584 | Better (misleading) |

## Critical Observations

### 1. Batching Wait is NOT the primary cause
- Disabling batching (batch_wait_ticks=0) still gives only 34 deliveries
- The problem exists even without the batching feature

### 2. Efficiency decreased significantly
- Distance per delivery: 18.2 → 23.2 (27% worse)
- This suggests either:
  - Worse routing/assignment decisions
  - Riders traveling longer for fewer orders
  - Sub-optimal batching/grouping

### 3. More orders left pending
- baseline_pau: 0 pending at end
- after_fusion: 9 pending at end (with fixed spawn)
- This suggests orders ARE being generated but NOT being assigned/delivered

### 4. Spawn cutoff was a bug but NOT the root cause
- Fixing spawn cutoff increased pending from 1 to 9
- But deliveries stayed at 36
- Therefore, the bottleneck is NOT order generation

## Hypotheses for Root Cause

### Hypothesis A: Different Assignment Engine Logic
**Evidence:**
- baseline_pau might have been run on `pau_intent` branch
- `pau_intent` and `rodolfo_intento` might have different `AssignmentEngine` implementations
- The "fusion" might have merged heuristic but kept rodolfo's assignment engine which could be less efficient

**Test:** Compare `core/assignment_engine.py` between branches

### Hypothesis B: Different Dispatch Policy
**Evidence:**
- `heuristic_benchmark.py` uses `choose_action()` which calls dispatch_policy actions
- If `pau_intent` had different action selection logic, results would differ

**Test:** Compare how actions are chosen in both branches

### Hypothesis C: Different Order Generation Parameters
**Evidence:**
- Spawn probability, max_eta, or other order generation params might differ
- Even with same seed, if underlying code differs, RNG sequence could diverge

**Test:** Add logging to track:
- Total orders generated
- Orders per tick
- Average order lifetime

### Hypothesis D: Rider Availability Logic Changed
**Evidence:**
- `get_available_riders()` now includes riders with `wait_until > 0`
- This might be excluding riders at critical moments
- Or making riders unavailable when they should be available

**Test:** Log rider states:
- Available count per tick
- Riders waiting per tick
- Assignment attempts vs successes

## Recommendation

**The comparison is INVALID because we're comparing different branches:**

1. `baseline_pau.json` was run on `pau_intent` branch (likely different implementation)
2. `after_fusion.json` was run on `rodolfo_intento` branch (post-merge)

**These are NOT comparable if the branches have different core logic beyond just heuristic.**

### What should have been done:

1. Checkout `rodolfo_intento` BEFORE merge
2. Run `python heuristic_benchmark.py --output baseline_rodolfo_pre.json [args]`
3. Merge pau_intent heuristic changes
4. Run `python heuristic_benchmark.py --output after_fusion.json [args]`
5. Compare baseline_rodolfo_pre vs after_fusion

**This would give a true before/after of THE SAME branch with ONLY the heuristic changes.**

## Next Steps

1. **CRITICAL:** Verify what branch baseline_pau.json was run on
2. If it was pau_intent, re-run baseline on rodolfo_intento pre-merge
3. If rodolfo_intento pre-merge ALSO gives ~36 deliveries, then no regression occurred
4. If rodolfo_intento pre-merge gives ~53 deliveries, then we have a real regression to investigate

## Code Analysis

### Files Changed in PR:
1. `core/fleet_manager.py` - Added `wait_until` field and logic to `get_available_riders()`
2. `simulation/simulator.py` - Added batching wait logic in `_rebuild_plan_for_rider()` and `move_riders_one_tick()`
3. `heuristic_benchmark.py` - NEW file for benchmarking (wasn't in either branch before)

### Potential Issues in Changes:

#### fleet_manager.py:68-80 - get_available_riders()
```python
def get_available_riders(self) -> List[Rider]:
    result: List[Rider] = []
    for r in self._riders:
        if r.resting:
            continue
        if r.available:
            result.append(r)
            continue
        if r.wait_until > 0 and r.can_take_more():  # ← NEW logic
            result.append(r)
    return result
```

**Analysis:** This adds riders who are waiting (wait_until > 0) to available list.
- **Potential issue:** A rider with wait_until > current_tick should NOT be available yet
- The code checks `wait_until > 0` but doesn't check `wait_until <= sim.t`
- This could make riders "available" even though they're supposed to be waiting

**FIX NEEDED:**
```python
if r.wait_until > 0 and r.wait_until <= current_tick and r.can_take_more():
```

But wait, the simulator doesn't pass current_tick to this function, so this check can't be done here.

#### simulator.py:490-498 - Wait check in move_riders_one_tick()
```python
if (
    tgt == self.restaurant
    and (not r.has_picked_up)
    and r.assigned_order_ids
    and r.wait_until > self.t  # ← Waiting until future tick
    and r.can_take_more()
):
    r.available = False
    continue
```

**Analysis:** This makes rider unavailable if still waiting. This looks correct.

**But:** The rider is in `get_available_riders()` because of line 78 in fleet_manager.py!

This is a RACE CONDITION:
1. Rider gets assigned orders and wait_until is set to future tick
2. `get_available_riders()` includes this rider because wait_until > 0
3. Assignment engine tries to assign MORE orders
4. But in `move_riders_one_tick()`, rider is blocked from moving

**This could cause:**
- Multiple assignment attempts to waiting riders
- Inefficient use of rider capacity
- Orders getting "stuck" on waiting riders who can't leave yet

## SMOKING GUN: get_available_riders() bug

**The bug in `core/fleet_manager.py:78`:**

```python
if r.wait_until > 0 and r.can_take_more():
    result.append(r)
```

This should be:
```python
# Don't include riders who are still waiting - they're not available for NEW assignments
# They already have orders and are waiting to batch them
```

Or if the intent is to allow batching to waiting riders, there needs to be coordination with simulator.

**This is likely causing riders to be marked as "available" when they're actually waiting, leading to:**
- Assignment attempts to riders who can't move yet
- Wasted assignment opportunities
- Orders not being assigned to truly available riders

