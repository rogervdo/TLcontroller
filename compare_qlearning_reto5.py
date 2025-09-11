"""
Comparison script to demonstrate Q-learning vs Fixed-time control in reto5.py
"""

import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))

import reto5
import agentpy as ap


def run_comparison_with_animation():
    """Run comparison between Q-learning and fixed-time control with animations
    Both methods now switch traffic lights every 10 steps for fair comparison
    """
    print("Running Q-Learning vs Fixed-Time Comparison with Animations...")
    print("Both methods switch traffic lights every 10 steps")
    print("=" * 60)

    # Common parameters
    base_params = {
        "steps": 100,
        "spawn_rate": 0.8,  # Higher spawn rate for more traffic
        "world_size": 500,
        "preset": "morning",
        "animation": True,  # Enable animation
    }

    results = {}

    # Test 1: Fixed-time control
    print("\n1. Running Fixed-Time Control with Animation...")
    params_fixed = base_params.copy()
    model_fixed = reto5.TrafficModel(params_fixed)
    model_fixed.enable_qlearning(False)  # Disable Q-learning
    model_fixed.setup()

    # Create animation for fixed-time control
    fig1, ax1 = plt.subplots(figsize=(10, 10))

    def animate_step_fixed(model, ax):
        reto5.draw_simulation(model, ax)

    # Run with animation
    anim_fixed = ap.animate(model_fixed, fig1, ax1, animate_step_fixed)

    # Save fixed-time animation
    print("Saving Fixed-Time animation...")
    anim_fixed.save("Traffic_FixedTime.gif", writer="pillow", fps=5)
    print("✓ Fixed-Time animation saved as 'Traffic_FixedTime.gif'")

    results["fixed"] = {
        "cars_completed": model_fixed.total_cars_arrived,
        "cars_spawned": model_fixed.total_cars_spawned,
        "completion_rate": model_fixed.total_cars_arrived
        / max(model_fixed.total_cars_spawned, 1),
    }

    print(f"✓ Fixed-time completed: {model_fixed.total_cars_arrived} cars arrived")

    # Test 2: Q-learning control
    print("\n2. Running Q-Learning Control with Animation...")
    params_ql = base_params.copy()
    model_ql = reto5.TrafficModel(params_ql)
    model_ql.enable_qlearning(True)  # Enable Q-learning
    model_ql.setup()

    # Create animation for Q-learning control
    fig2, ax2 = plt.subplots(figsize=(10, 10))

    def animate_step_ql(model, ax):
        reto5.draw_simulation(model, ax)

    # Run with animation
    anim_ql = ap.animate(model_ql, fig2, ax2, animate_step_ql)

    # Save Q-learning animation
    print("Saving Q-Learning animation...")
    anim_ql.save("Traffic_QLearning.gif", writer="pillow", fps=5)
    print("✓ Q-Learning animation saved as 'Traffic_QLearning.gif'")

    results["ql"] = {
        "cars_completed": model_ql.total_cars_arrived,
        "cars_spawned": model_ql.total_cars_spawned,
        "completion_rate": model_ql.total_cars_arrived
        / max(model_ql.total_cars_spawned, 1),
    }

    print(f"✓ Q-learning completed: {model_ql.total_cars_arrived} cars arrived")

    # Print comparison results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    fixed_completion = results["fixed"]["cars_completed"]
    ql_completion = results["ql"]["cars_completed"]

    print("Fixed-Time Control:")
    print(f"  Cars completed: {fixed_completion}")
    print(f"  Completion rate: {results['fixed']['completion_rate']:.1%}")

    print("\nQ-Learning Control:")
    print(f"  Cars completed: {ql_completion}")
    print(f"  Completion rate: {results['ql']['completion_rate']:.1%}")

    improvement = ((ql_completion - fixed_completion) / max(fixed_completion, 1)) * 100
    print(f"\nImprovement: {improvement:+.1f}% more cars completed with Q-learning")

    # Get Q-learning stats
    if hasattr(model_ql, "get_ql_stats"):
        ql_stats = model_ql.get_ql_stats()
        print("\nQ-Learning Details:")
        print(f"  Q-table size: {ql_stats['q_table_size']}")
        print(f"  Final epsilon: {ql_stats['epsilon']:.3f}")

    print("\n" + "=" * 60)
    print("ANIMATION FILES SAVED:")
    print("  Fixed-Time: Traffic_FixedTime.gif")
    print("  Q-Learning: Traffic_QLearning.gif")
    print("=" * 60)

    return results, anim_fixed, anim_ql


if __name__ == "__main__":
    results, anim_fixed, anim_ql = run_comparison_with_animation()
    print("\n✓ Comparison with animations completed successfully!")
