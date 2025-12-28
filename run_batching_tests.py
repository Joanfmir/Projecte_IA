#!/usr/bin/env python
"""Run batching tests."""
import sys
sys.path.insert(0, '.')

from tests.test_batching_logic import (
    test_batching_prefers_partial_rider_cluster,
    test_wait_updates_q_value_on_backlog
)

if __name__ == "__main__":
    print("Running batching tests...")
    print("\n" + "="*60)
    print("Test 1: test_batching_prefers_partial_rider_cluster")
    print("="*60)
    try:
        test_batching_prefers_partial_rider_cluster()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Test 2: test_wait_updates_q_value_on_backlog")
    print("="*60)
    try:
        test_wait_updates_q_value_on_backlog()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*60)
    print("All batching tests PASSED! ✓")
    print("="*60)
