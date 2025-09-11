"""
Test fixed-time switching in reto5.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import reto5

def test_fixed_time_switching():
    """Test that fixed-time switching works when Q-learning is disabled"""
    print("Testing Fixed-Time Switching in reto5.py...")
    print("=" * 50)

    # Create a test model
    test_params = {
        "steps": 50,  # Run for 50 steps to see multiple switches
        "spawn_rate": 0.1,
        "world_size": 500,
        "preset": "morning",
        "animation": False,
    }

    model = reto5.TrafficModel(test_params)
    model.enable_qlearning(False)  # Disable Q-learning to test fixed-time
    model.setup()

    print(f"✓ Q-learning disabled: {not model.ql_enabled}")
    print(f"✓ Initial active group: {model.active_group}")

    # Track group changes
    group_changes = []

    print("\nRunning simulation and tracking group changes...")
    for i in range(1, test_params["steps"] + 1):
        model.t = i
        old_group = model.active_group
        model.step()
        new_group = model.active_group

        if old_group != new_group:
            group_changes.append((i, old_group, new_group))
            print(f"Step {i}: Group {old_group} → Group {new_group}")

    print(f"\n✓ Total group changes: {len(group_changes)}")
    print("Expected changes at steps: 10, 20, 30, 40, 50")
    print(f"Actual changes at steps: {[change[0] for change in group_changes]}")

    # Verify the changes happened at the right times
    expected_steps = [10, 20, 30, 40, 50]
    actual_steps = [change[0] for change in group_changes]

    if actual_steps == expected_steps:
        print("✓ Fixed-time switching works correctly!")
        return True
    else:
        print("✗ Fixed-time switching is not working as expected")
        return False

if __name__ == "__main__":
    test_fixed_time_switching()
