#!/usr/bin/env python
"""Run batching tests."""
import sys
import traceback

sys.path.insert(0, '.')

from tests.test_batching_logic import (
    test_batching_prefers_partial_rider_cluster,
    test_wait_updates_q_value_on_backlog
)


def run_test(test_name, test_func):
    """Helper function to run a test and handle exceptions."""
    print("\n" + "="*60)
    print(f"Test: {test_name}")
    print("="*60)
    try:
        test_func()
        print("✓ PASSED")
        return True
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running batching tests...")
    
    tests = [
        ("test_batching_prefers_partial_rider_cluster", test_batching_prefers_partial_rider_cluster),
        ("test_wait_updates_q_value_on_backlog", test_wait_updates_q_value_on_backlog),
    ]
    
    results = [run_test(name, func) for name, func in tests]
    
    if all(results):
        print("\n" + "="*60)
        print("All batching tests PASSED! ✓")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("Some tests FAILED! ✗")
        print("="*60)
        sys.exit(1)
