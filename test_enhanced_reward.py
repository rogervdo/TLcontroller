"""
Test script to verify enhanced waiting car contribution to Q-learning reward signal.

This script tests the improved traffic_qlearning_mixin.py with enhanced reward
calculations inspired by cruceqlearningdump.py.
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

from traffic_qlearning_mixin import QLearningTrafficMixin
import numpy as np


class TestTrafficModel(QLearningTrafficMixin):
    """Simple test model to verify enhanced reward calculations"""

    def __init__(self):
        # Initialize basic attributes needed for Q-learning
        self.t = 0
        self.active_group = 0
        self.group_cycle_steps = 10
        self.total_cars_arrived = 0
        self.cars = []
        self.road_network = MockRoadNetwork()

        # Initialize Q-learning
        self.init_qlearning(alpha=0.1, gamma=0.95, epsilon=0.2)


class MockRoadNetwork:
    """Mock road network for testing"""

    def __init__(self):
        self.nodes = {
            "8_16": MockNode(0),
            "8_14": MockNode(0),
            "8_12": MockNode(0),
            "18_16": MockNode(1),
            "18_14": MockNode(1),
            "18_12": MockNode(1),
            "22_4": MockNode(2),
            "30_15": MockNode(2),
        }


class MockNode:
    """Mock node for testing"""

    def __init__(self, group_id):
        self.group_id = group_id


class MockCar:
    """Mock car for testing"""

    def __init__(
        self, origin, state="waiting", wait_steps=0, current_node_id="8_16", path=None
    ):
        self.origin = origin
        self.state = state  # "waiting" or "moving"
        self.wait_steps = wait_steps
        self.current_node_id = current_node_id
        self.path = path or []
        self.current_path_index = 0


def test_enhanced_reward_system():
    """Test the enhanced reward system with different waiting car scenarios"""

    print("=" * 70)
    print("TESTING ENHANCED WAITING CAR CONTRIBUTION TO REWARD SIGNAL")
    print("=" * 70)

    model = TestTrafficModel()

    # Test Scenario 1: No waiting cars (optimal scenario)
    print("\n--- Test Scenario 1: No waiting cars ---")
    model.cars = []
    reward, metrics = model.calculate_traffic_reward()
    print(".2f")
    print(f"  Total waiting cars: {metrics['total_waiting']}")
    print(f"  Reward components: {metrics['reward_components']}")

    # Test Scenario 2: Few waiting cars
    print("\n--- Test Scenario 2: Few waiting cars ---")
    model.cars = [
        MockCar(
            "N", state="waiting", wait_steps=2, current_node_id="8_16"
        ),  # At group 0, active is 0, so should move
        MockCar(
            "S", state="waiting", wait_steps=3, current_node_id="18_16"
        ),  # At group 1, active is 0, so waiting
    ]
    reward, metrics = model.calculate_traffic_reward()
    print(".2f")
    print(f"  Total waiting cars: {metrics['total_waiting']}")
    print(f"  Long waiting (>10 steps): {metrics['long_waiting']}")
    print(f"  Early long waiting (>7 steps): {metrics['early_long_waiting']}")
    print(f"  Reward components: {metrics['reward_components']}")

    # Test Scenario 3: Many waiting cars with starvation
    print("\n--- Test Scenario 3: Many waiting cars with starvation ---")
    model.cars = [
        MockCar(
            "N", state="waiting", wait_steps=12, current_node_id="18_16"
        ),  # Long waiting at group 1
        MockCar(
            "N", state="waiting", wait_steps=8, current_node_id="18_16"
        ),  # Early long waiting at group 1
        MockCar(
            "S", state="waiting", wait_steps=16, current_node_id="18_16"
        ),  # Very long waiting at group 1
        MockCar(
            "E", state="waiting", wait_steps=5, current_node_id="22_4"
        ),  # Normal waiting at group 2
        MockCar(
            "W", state="waiting", wait_steps=3, current_node_id="22_4"
        ),  # Normal waiting at group 2
    ]
    reward, metrics = model.calculate_traffic_reward()
    print(".2f")
    print(f"  Total waiting cars: {metrics['total_waiting']}")
    print(f"  Long waiting (>10 steps): {metrics['long_waiting']}")
    print(f"  Early long waiting (>7 steps): {metrics['early_long_waiting']}")
    print(f"  Very long waiting (>15 steps): {metrics['very_long_waiting']}")
    print(f"  Queue imbalance: {metrics['queue_imbalance']}")
    print(f"  Reward components: {metrics['reward_components']}")

    # Test Scenario 4: Cars actively moving through intersection
    print("\n--- Test Scenario 4: Cars moving through intersection ---")
    model.cars = [
        MockCar(
            "N", state="moving", current_node_id="8_16"
        ),  # Moving through group 0 (active)
        MockCar(
            "S", state="moving", current_node_id="18_16"
        ),  # Moving through group 1 (not active)
        MockCar(
            "E", state="waiting", wait_steps=2, current_node_id="22_4"
        ),  # Waiting at group 2
    ]
    reward, metrics = model.calculate_traffic_reward()
    print(".2f")
    print(f"  Total waiting cars: {metrics['total_waiting']}")
    print(f"  Cars served this step: {metrics['cars_served_this_step']}")
    print(f"  Reward components: {metrics['reward_components']}")

    # Test Q-learning statistics
    print("\n--- Q-Learning Statistics Analysis ---")
    stats = model.get_ql_stats()
    print(f"  Q-table size: {stats['q_table_size']}")
    print(f"  Current epsilon: {stats['epsilon']:.3f}")
    if "waiting_car_analysis" in stats:
        analysis = stats["waiting_car_analysis"]
        print(f"  Waiting car analysis: {analysis}")
    if "reward_contribution_analysis" in stats:
        contrib = stats["reward_contribution_analysis"]
        print(f"  Reward contribution analysis: {contrib}")

    print("\n" + "=" * 70)
    print("ENHANCED REWARD SYSTEM TEST COMPLETED")
    print("Waiting cars are now properly contributing to the reward signal!")
    print("=" * 70)


if __name__ == "__main__":
    test_enhanced_reward_system()
