"""
Test script to verify Q-learning mixin integration

This script creates a minimal test to ensure the Q-learning mixin
can be properly integrated with the traffic simulation.
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

from traffic_qlearning_mixin import QLearningTrafficMixin
import agentpy as ap


class TestTrafficModel(ap.Model, QLearningTrafficMixin):
    """Minimal test model to verify Q-learning integration"""

    def setup(self):
        # Minimal setup for testing
        self.active_group = 0
        self.group_cycle_steps = 5
        self.t = 0
        self.cars = []  # Empty car list for testing
        self.total_cars_arrived = 0
        self.total_cars_spawned = 0

        # Initialize Q-learning
        self.init_qlearning(
            alpha=0.1, gamma=0.95, epsilon=0.2, epsilon_decay=0.995, min_epsilon=0.05
        )

        print("✓ Q-Learning mixin successfully integrated")

    def step(self):
        self.t += 1

        # Test Q-learning step
        if self.t % 5 == 0:
            self.ql_step()

        if self.t >= 10:  # Stop after 10 steps for testing
            self.stop()


def test_qlearning_integration():
    """Test the Q-learning mixin integration"""
    print("Testing Q-Learning Mixin Integration...")
    print("=" * 50)

    # Create test model
    model = TestTrafficModel({"steps": 10})

    # Run test
    model.run()

    # Check Q-learning functionality
    stats = model.get_ql_stats()
    print("\nQ-Learning Test Results:")
    print(f"✓ Q-table size: {stats['q_table_size']}")
    print(f"✓ Final epsilon: {stats['epsilon']:.3f}")
    print(f"✓ Total reward: {stats['total_reward']:.3f}")

    print("\n✓ Q-Learning mixin integration test completed successfully!")
    return True


if __name__ == "__main__":
    test_qlearning_integration()
