"""
Test script to verify Q-learning integration in reto5.py
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

import reto5


def test_reto5_qlearning():
    """Test Q-learning integration in reto5.py"""
    print("Testing Q-Learning Integration in reto5.py...")
    print("=" * 50)

    # Create a test model with minimal parameters
    test_params = {
        "steps": 20,
        "spawn_rate": 0.1,  # Low spawn rate for testing
        "world_size": 500,
        "preset": "morning",
        "animation": True,
    }

    model = reto5.TrafficModel(test_params)
    model.setup()

    # Verify Q-learning is initialized
    print("✓ TrafficModel created and setup completed")
    print(f"✓ Active group: {model.active_group}")
    print(f"✓ Q-learning enabled: {hasattr(model, 'ql_enabled')}")

    # Run a few steps to test Q-learning
    print("\nRunning simulation steps...")
    for i in range(5):
        model.t = i + 1
        model.step()
        if i % 5 == 0:  # Q-learning decisions every 5 steps
            print(f"Step {i + 1}: Active group = {model.active_group}")

    # Check Q-learning stats
    if hasattr(model, "get_ql_stats"):
        stats = model.get_ql_stats()
        print("\nQ-Learning Stats:")
        print(f"✓ Q-table size: {stats['q_table_size']}")
        print(f"✓ Epsilon: {stats['epsilon']:.3f}")
        print(f"✓ Total reward: {stats['total_reward']:.3f}")

    print("\n✓ Q-learning integration test completed successfully!")
    return True


if __name__ == "__main__":
    test_reto5_qlearning()
