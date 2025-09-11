"""
Enhanced Comparison Study for reto5.py Traffic Simulation

This script implements the enhanced comparison study as a mixin for reto5.py,
providing comprehensive analysis between Q-learning and fixed-time traffic controllers.

Usage:
    python compare_qlearning_reto5.py

Features:
- Automated comparison between Q-learning and fixed-time controllers
- Performance metrics analysis (waiting time, throughput, completion rate)
- Visualization of comparison results
- Statistical analysis of improvements
"""

import sys
import os
import numpy as np

# Add current directory to path to import reto5
sys.path.append(os.path.dirname(__file__))

# Import reto5 and the mixins
import reto5
from enhanced_comparison_mixin import EnhancedComparisonMixin
from traffic_qlearning_mixin import QLearningTrafficMixin


class EnhancedTrafficModel(
    reto5.TrafficModel, EnhancedComparisonMixin, QLearningTrafficMixin
):
    """
    Extended TrafficModel that inherits from reto5.TrafficModel and adds
    enhanced comparison study functionality via mixin.
    """

    def __init__(self, parameters):
        # Initialize with reto5.TrafficModel's parameters
        super().__init__(parameters)

    def get_comparison_stats(self):
        """
        Get comprehensive statistics for comparison analysis.
        Enhanced version that works with reto5's data structures.
        """
        # Calculate average waiting time from completed cars
        if (
            hasattr(self, "completed_cars_wait_times")
            and self.completed_cars_wait_times
        ):
            avg_wait = np.mean(self.completed_cars_wait_times)
        else:
            # Fallback: estimate from current cars
            if len(self.cars) > 0:
                avg_wait = np.mean([getattr(car, "wait_steps", 0) for car in self.cars])
            else:
                avg_wait = 0

        # Calculate throughput (cars per second)
        throughput = self.total_cars_arrived / max(self.t, 1)

        # Completion rate
        completion_rate = self.total_cars_arrived / max(self.total_cars_spawned, 1)

        base_stats = {
            "avg_waiting_time": avg_wait,
            "throughput": throughput,
            "completion_rate": completion_rate,
            "total_cars_spawned": self.total_cars_spawned,
            "total_cars_arrived": self.total_cars_arrived,
            "controller_type": "Q-Learning"
            if getattr(self, "ql_enabled", False)
            else "Fixed-Time",
        }

        # Add Q-learning specific stats if available
        if hasattr(self, "ql_enabled") and self.ql_enabled:
            ql_stats = self.get_ql_stats()
            base_stats.update(
                {
                    "total_reward": ql_stats.get("total_reward", 0),
                    "q_table_size": ql_stats.get("q_table_size", 0),
                    "epsilon": ql_stats.get("epsilon", 0),
                    "performance_history": ql_stats.get("performance_history", []),
                }
            )

        return base_stats


def run_enhanced_comparison_study_with_reto5(params, steps=800):
    """
    Run enhanced comparison study using reto5's TrafficModel with mixin functionality.

    Args:
        params: Dictionary of simulation parameters
        steps: Number of simulation steps to run

    Returns:
        tuple: (models_dict, results_dict) containing simulation results
    """
    print("Starting Enhanced Comparison Study with reto5.py...")
    print("=" * 60)

    results = {}
    models = {}

    # Test scenarios
    scenarios = [
        {"use_qlearning": False, "name": "Fixed-Time"},
        {"use_qlearning": True, "name": "Q-Learning"},
    ]

    for i, scenario in enumerate(scenarios):
        print(f"\n[{i + 1}/2] Running {scenario['name']} Controller...")

        # Configure parameters
        test_params = params.copy()
        test_params.update({"use_qlearning": scenario["use_qlearning"], "steps": steps})

        # Create and run model
        model = EnhancedTrafficModel(test_params)

        # Enable/disable Q-learning as needed
        if scenario["use_qlearning"]:
            model.ql_enabled = True
            # Initialize Q-learning parameters if not already done
            if not hasattr(model, "Q"):
                model.init_qlearning()
            print("Q-Learning enabled - exclusive traffic light control activated")
            print(
                f"Q-Learning parameters: α={model.ql_alpha}, γ={model.ql_gamma}, ε={model.ql_epsilon}"
            )
        else:
            model.ql_enabled = False
            print("Q-Learning disabled - using fixed-time control")

        # Run simulation
        model.run()

        # Store results
        stats = model.get_comparison_stats()
        results[scenario["name"]] = stats
        models[scenario["name"]] = model

        print(f"  ✓ Completed {scenario['name']} simulation")
        print(".2f")
        print(".1f")
        print(".1%")

    # Calculate improvements
    ql_stats = results["Q-Learning"]
    fixed_stats = results["Fixed-Time"]

    waiting_improvement = (
        (fixed_stats["avg_waiting_time"] - ql_stats["avg_waiting_time"])
        / max(fixed_stats["avg_waiting_time"], 0.001)
    ) * 100
    throughput_improvement = (
        (ql_stats["throughput"] - fixed_stats["throughput"])
        / max(fixed_stats["throughput"], 0.001)
    ) * 100

    print("\n" + "=" * 60)
    print("COMPARATIVE PERFORMANCE ANALYSIS")
    print("=" * 60)
    print("Q-Learning vs Fixed-Time Performance:")
    print(".1f")
    print(".1f")

    if waiting_improvement > 5:
        print(".1f")
    elif waiting_improvement > 0:
        print(".1f")
    else:
        print(".1f")

    if throughput_improvement > 5:
        print(".1f")
    elif throughput_improvement > 0:
        print(".1f")
    else:
        print(".1f")

    # Generate individual analysis for each controller
    print("\n" + "=" * 60)
    print("INDIVIDUAL CONTROLLER ANALYSIS")
    print("=" * 60)

    # Create a temporary model instance to access mixin methods
    temp_model = EnhancedTrafficModel(params)

    # Analyze Fixed-Time Controller
    fixed_model_copy = models["Fixed-Time"]
    print(
        f"Controller Type: {fixed_model_copy.get_comparison_stats()['controller_type']}"
    )
    temp_model.improved_analysis_function(
        fixed_model_copy,
        {"use_qlearning": False, "description": "reto5.py simulation"},
        " - Fixed-Time",
    )

    # Analyze Q-Learning Controller
    ql_model_copy = models["Q-Learning"]
    print(f"Controller Type: {ql_model_copy.get_comparison_stats()['controller_type']}")
    temp_model.improved_analysis_function(
        ql_model_copy,
        {"use_qlearning": True, "description": "reto5.py simulation"},
        " - Q-Learning",
    )

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    temp_model.plot_comparison_results(
        models, save_path="traffic_comparison_results.png"
    )

    print("\n✓ Comparison study completed successfully!")
    return models, results


def main():
    """Main function to run the enhanced comparison study with reto5.py"""
    print("=" * 70)
    print("ENHANCED COMPARISON STUDY FOR RETO5.PY TRAFFIC SIMULATION")
    print("=" * 70)
    print("This script compares Q-learning vs Fixed-time traffic controllers")
    print("using reto5.py with enhanced comparison mixin functionality.")
    print()

    # Simulation parameters (compatible with reto5.py)
    params = {
        "steps": 200,  # Use the full 2000 steps as defined in reto5.py
        "spawn_rate": 0.3,
        "world_size": 500,
        "preset": "morning",
        "animation": True,  # Disable animation for faster comparison
    }

    # Run the enhanced comparison study
    print("Starting comparison study...")
    models_dict, results_dict = run_enhanced_comparison_study_with_reto5(
        params,
        steps=500,  # Use full 2000 steps for comprehensive analysis
    )

    print("\n" + "=" * 70)
    print("COMPARISON STUDY COMPLETED")
    print("=" * 70)
    print("Results summary:")
    for controller, stats in results_dict.items():
        print(f"  {controller}:")
        print(".2f")
        print(".1f")
        print(".1%")

    return models_dict, results_dict


if __name__ == "__main__":
    # Run the comparison study
    models, results = main()
