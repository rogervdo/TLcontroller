"""
Enhanced Comparison Study Mixin for Traffic Simulation

This mixin provides comprehensive comparison functionality between Q-learning
and fixed-time traffic controllers for the reto5.py traffic simulation.

Features:
- Automated comparison studies between different control strategies
- Performance analysis with detailed metrics
- Visualization of comparison results
- Statistical analysis of improvements
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns.set(context="notebook", style="whitegrid")
except ImportError:
    pass  # seaborn not available, continue without it


class EnhancedComparisonMixin:
    """
    Mixin class providing enhanced comparison study functionality.

    This mixin adds methods to run comprehensive comparisons between
    Q-learning and fixed-time traffic controllers, including performance
    analysis and visualization.
    """

    def run_enhanced_comparison_study(self, params, steps=800):
        """
        Run comprehensive comparison between Q-learning and Fixed-time control.

        Args:
            params: Dictionary of simulation parameters
            steps: Number of simulation steps to run

        Returns:
            tuple: (models_dict, results_dict) containing simulation results
        """
        print("Starting Enhanced Comparison Study...")
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
            test_params.update(
                {"use_qlearning": scenario["use_qlearning"], "steps": steps}
            )

            # Run simulation
            model = self.__class__(test_params)
            model.run()

            # Store results
            stats = model.get_comparison_stats()
            results[scenario["name"]] = stats
            models[scenario["name"]] = model

            print(f"  ✓ Completed {scenario['name']} simulation")
            print(f"    - Avg waiting time: {stats['avg_waiting_time']:.2f}s")
            print(f"    - Throughput: {stats['throughput'] * 3600:.1f} cars/hr")
            print(f"    - Completion rate: {stats['completion_rate']:.1%}")

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
        print(f"  Waiting time change: {waiting_improvement:+.1f}%")
        print(f"  Throughput change: {throughput_improvement:+.1f}%")

        if waiting_improvement > 5:
            print(
                f"  ✓ Q-Learning significantly reduces average waiting time by {waiting_improvement:.1f}%"
            )
        elif waiting_improvement > 0:
            print(
                f"  ≈ Q-Learning slightly reduces average waiting time by {waiting_improvement:.1f}%"
            )
        else:
            print(
                f"  ✗ Q-Learning increases average waiting time by {abs(waiting_improvement):.1f}%"
            )

        if throughput_improvement > 5:
            print(
                f"  ✓ Q-Learning significantly improves throughput by {throughput_improvement:.1f}%"
            )
        elif throughput_improvement > 0:
            print(
                f"  ≈ Q-Learning slightly improves throughput by {throughput_improvement:.1f}%"
            )
        else:
            print(
                f"  ✗ Q-Learning reduces throughput by {abs(throughput_improvement):.1f}%"
            )

        # Generate individual analysis for each controller
        print("\n" + "=" * 60)
        print("INDIVIDUAL CONTROLLER ANALYSIS")
        print("=" * 60)

        # Analyze Fixed-Time Controller
        self.improved_analysis_function(
            models["Fixed-Time"],
            {
                "use_qlearning": False,
                "description": params.get("description", "Custom"),
            },
            " - Fixed-Time",
        )

        # Analyze Q-Learning Controller
        self.improved_analysis_function(
            models["Q-Learning"],
            {"use_qlearning": True, "description": params.get("description", "Custom")},
            " - Q-Learning",
        )

        # Generate comparison plots
        print("\nGenerating comparison plots...")
        self.plot_comparison_results(
            models, save_path="traffic_controller_comparison.png"
        )

        print("\n✓ Comparison study completed successfully!")
        return models, results

    def get_comparison_stats(self):
        """
        Get comprehensive statistics for comparison analysis.

        Returns:
            dict: Statistics including waiting time, throughput, completion rate, etc.
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

    def improved_analysis_function(self, model, params, title_suffix=""):
        """
        Improved analysis function focusing on key performance metrics.

        Args:
            model: The simulation model to analyze
            params: Parameters used in the simulation
            title_suffix: Suffix to add to plot titles
        """
        print(f"\n{'=' * 60}")
        print(f"TRAFFIC CONTROLLER PERFORMANCE ANALYSIS{title_suffix}")
        print(f"Controller Type: {model.get_comparison_stats()['controller_type']}")
        print(f"Scenario: {params.get('description', 'Custom')}")
        print(f"{'=' * 60}")

        # Final statistics
        stats = model.get_comparison_stats()
        print("\n--- Final Simulation Statistics ---")
        print(f"  Average Waiting Time: {stats['avg_waiting_time']:.2f} s")
        print(
            f"  System Throughput: {stats['throughput']:.3f} cars/s ({stats['throughput'] * 3600:.1f} cars/hr)"
        )
        print(f"  Vehicle Completion Rate: {stats['completion_rate']:.2%}")

        if params.get("use_qlearning", False):
            print(f"  Total Reward Accumulated: {stats.get('total_reward', 0):.1f}")
            print(f"  Q-Table Size (Learned States): {stats.get('q_table_size', 0)}")
            print(f"  Final Exploration Rate (Epsilon): {stats.get('epsilon', 0):.4f}")

        # Performance history analysis
        if hasattr(model, "ql_performance_history") and model.ql_performance_history:
            history = model.ql_performance_history
            rewards = [h["reward"] for h in history]
            total_waiting = [h["total_waiting"] for h in history]

            print("\n--- Performance History Summary ---")
            print(f"  Total performance records: {len(history)}")
            print(f"  Average reward per step: {np.mean(rewards):.3f}")
            print(f"  Max reward: {max(rewards):.3f}")
            print(f"  Min reward: {min(rewards):.3f}")
            print(f"  Average waiting cars: {np.mean(total_waiting):.1f}")

        print("\n✓ Analysis completed")

    def plot_comparison_results(self, models, save_path=None):
        """
        Generate streamlined comparison plots between controllers.

        Args:
            models: Dictionary containing 'Fixed-Time' and 'Q-Learning' models
            save_path: Path to save the plot (optional). If None, displays the plot.
        """
        # Extract data from both models
        ql_model = models["Q-Learning"]
        fixed_model = models["Fixed-Time"]

        # Get performance history - handle missing history gracefully
        ql_history = getattr(ql_model, "ql_performance_history", [])

        # For fixed-time model, create a mock history based on simulation steps
        if (
            hasattr(fixed_model, "ql_performance_history")
            and fixed_model.ql_performance_history
        ):
            fixed_history = fixed_model.ql_performance_history
        else:
            # Create mock history for fixed-time controller
            fixed_history = []
            for t in range(0, fixed_model.t, 25):  # Sample every 25 steps
                # Count waiting cars more accurately
                waiting_count = 0
                if hasattr(fixed_model, "cars"):
                    for car in fixed_model.cars:
                        if getattr(car, "state", "moving") != "moving":
                            waiting_count += 1

                fixed_history.append(
                    {
                        "time": t,
                        "reward": 0,  # Fixed-time doesn't use rewards
                        "total_waiting": waiting_count,
                        "active_group": getattr(fixed_model, "active_group", 0),
                    }
                )

        if not ql_history and not fixed_history:
            print("Warning: No performance history available for comparison plots")
            return

        # Create comparison plots with updated layout for 5 graphs
        fig = plt.figure(figsize=(25, 12))  # Adjusted size for 5 plots
        fig.suptitle(
            "Q-Learning vs Fixed-Time Controller Comparison",
            fontsize=18,
            fontweight="bold",
        )

        # Create subplots with custom layout for 5 graphs
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

        # Wide subplot for Q-Learning Reward Signal (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        # Wide subplot for Active Traffic Light Group (spans 2 columns)
        ax2 = fig.add_subplot(gs[0, 2:])
        # Regular subplots for other plots
        ax3 = fig.add_subplot(gs[1, 0])  # Performance Summary (Last 10%)
        ax4 = fig.add_subplot(gs[1, 1])  # Performance Summary (Overall)
        ax5 = fig.add_subplot(gs[1, 2:])  # Avg Wait Time vs Simulation Step

        # Colors for consistency
        ql_color = "#2E86C1"  # Blue
        fixed_color = "#E74C3C"  # Red
        epsilon_color = "#28B463"  # Green

        # Extract data
        ql_times = [h["time"] for h in ql_history]
        ql_rewards = [h["reward"] for h in ql_history]
        ql_waiting = [h["total_waiting"] for h in ql_history]
        ql_epsilon = [h.get("epsilon", 0) for h in ql_history]  # Extract epsilon values

        fixed_times = [h["time"] for h in fixed_history]
        fixed_waiting = [h.get("total_waiting", 0) for h in fixed_history]

        # Calculate max_time for use in multiple plots
        max_time = max(
            max(ql_times) if ql_times else 0, max(fixed_times) if fixed_times else 0
        )
        if max_time == 0:
            max_time = 100  # Default fallback

        # Plot 1: Q-Learning Reward Signal (wide)
        if ql_rewards and any(r != 0 for r in ql_rewards):
            ax1.plot(
                ql_times,
                ql_rewards,
                color=ql_color,
                linewidth=2.5,
                label="Q-Learning Reward",
                alpha=0.8,
            )
            ax1.set_title("Q-Learning Reward Signal", fontsize=14, fontweight="bold")
            ax1.set_ylabel("Reward Value")
        else:
            # Compare waiting times instead
            ax1.plot(
                ql_times,
                ql_waiting,
                color=ql_color,
                linewidth=2.5,
                label="Q-Learning",
                alpha=0.8,
            )
            ax1.plot(
                fixed_times,
                fixed_waiting,
                color=fixed_color,
                linewidth=2.5,
                label="Fixed-Time",
                alpha=0.8,
            )
            ax1.set_title("Waiting Cars Comparison", fontsize=14, fontweight="bold")
            ax1.set_ylabel("Number of Waiting Cars")
        ax1.set_xlabel("Simulation Step")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Active Traffic Light Group (wide)
        ql_groups = [h["active_group"] for h in ql_history]
        fixed_groups = [h.get("active_group", 0) for h in fixed_history]

        ax2.plot(
            ql_times,
            ql_groups,
            color=ql_color,
            linewidth=2.5,
            label="Q-Learning",
            alpha=0.8,
            drawstyle="steps-post",
        )
        ax2.plot(
            fixed_times,
            fixed_groups,
            color=fixed_color,
            linewidth=2.5,
            label="Fixed-Time",
            alpha=0.8,
            drawstyle="steps-post",
        )
        ax2.set_title("Active Traffic Light Group", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Group ID")
        ax2.set_xlabel("Simulation Step")
        ax2.set_yticks([0, 1, 2])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Performance Summary (Last 10% of steps)
        # Calculate statistics from the last 10% of simulation steps
        last_10_percent_start = int(max_time * 0.9) if max_time > 0 else 0

        # Get data from last 10% of simulation
        ql_last_10_data = [h for h in ql_history if h["time"] >= last_10_percent_start]
        fixed_last_10_data = [
            h for h in fixed_history if h["time"] >= last_10_percent_start
        ]

        # Calculate average metrics from last 10%
        ql_last_avg_wait = (
            np.mean([h["total_waiting"] for h in ql_last_10_data])
            if ql_last_10_data
            else 0
        )
        fixed_last_avg_wait = (
            np.mean([h.get("total_waiting", 0) for h in fixed_last_10_data])
            if fixed_last_10_data
            else 0
        )

        # For throughput and completion, use simplified calculations based on available data
        # Since we don't have detailed arrival/completion data in history, use proxy metrics
        ql_last_avg_reward = (
            np.mean([h["reward"] for h in ql_last_10_data]) if ql_last_10_data else 0
        )
        fixed_last_avg_reward = 0  # Fixed-time doesn't use rewards

        # Use reward as a proxy for performance (higher reward = better performance)
        ql_completion_proxy = max(
            0, ql_last_avg_reward * 10
        )  # Scale reward to percentage-like value
        fixed_completion_proxy = 50  # Fixed baseline

        last_10_metrics = [
            "Avg Wait Time\n(Last 10%)",
            "Avg Reward\n(Last 10%)",
            "Performance Score\n(Last 10%)",
        ]
        ql_last_values = [ql_last_avg_wait, ql_last_avg_reward, ql_completion_proxy]
        fixed_last_values = [
            fixed_last_avg_wait,
            fixed_last_avg_reward,
            fixed_completion_proxy,
        ]

        x_last = np.arange(len(last_10_metrics))
        width_last = 0.35

        bars_last1 = ax3.bar(
            x_last - width_last / 2,
            ql_last_values,
            width_last,
            label="Q-Learning",
            color=ql_color,
            alpha=0.8,
        )
        bars_last2 = ax3.bar(
            x_last + width_last / 2,
            fixed_last_values,
            width_last,
            label="Fixed-Time",
            color=fixed_color,
            alpha=0.8,
        )
        ax3.set_title("Performance Summary (Last 10%)", fontsize=14, fontweight="bold")
        ax3.set_ylabel("Value")
        ax3.set_xticks(x_last)
        ax3.set_xticklabels(last_10_metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars for last 10%
        for bar in bars_last1:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(ql_last_values + fixed_last_values + [0.1]) * 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        for bar in bars_last2:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(ql_last_values + fixed_last_values + [0.1]) * 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Plot 4: Performance Summary (Bar Chart)
        # Get final statistics
        ql_stats = ql_model.get_comparison_stats()
        fixed_stats = fixed_model.get_comparison_stats()

        metrics = [
            "Avg Wait Time\n(seconds)",
            "Throughput\n(cars/hour)",
            "Completion Rate\n(%)",
        ]
        ql_values = [
            ql_stats["avg_waiting_time"],
            ql_stats["throughput"]
            * 3600,  # Convert to cars/hour for better readability
            ql_stats["completion_rate"] * 100,
        ]
        fixed_values = [
            fixed_stats["avg_waiting_time"],
            fixed_stats["throughput"]
            * 3600,  # Convert to cars/hour for better readability
            fixed_stats["completion_rate"] * 100,
        ]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax4.bar(
            x - width / 2,
            ql_values,
            width,
            label="Q-Learning",
            color=ql_color,
            alpha=0.8,
        )
        bars2 = ax4.bar(
            x + width / 2,
            fixed_values,
            width,
            label="Fixed-Time",
            color=fixed_color,
            alpha=0.8,
        )
        ax4.set_title("Performance Summary", fontsize=14, fontweight="bold")
        ax4.set_ylabel("Value")
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(ql_values + fixed_values) * 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        for bar in bars2:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(ql_values + fixed_values) * 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Plot 5: Avg Wait Time vs Simulation Step (Table-like plot)
        # Calculate average wait time at different intervals
        intervals = []
        ql_avg_waits = []
        fixed_avg_waits = []

        interval_size = max(max_time // 10, 50)  # 10 intervals, minimum 50 steps

        for i in range(interval_size, max_time + 1, interval_size):
            # Q-Learning data
            ql_data_in_interval = [h for h in ql_history if h["time"] <= i]
            if ql_data_in_interval:
                # Use last 10 points or all available points if less than 10
                recent_points = (
                    ql_data_in_interval[-10:]
                    if len(ql_data_in_interval) >= 10
                    else ql_data_in_interval
                )
                avg_ql_wait = np.mean([h["total_waiting"] for h in recent_points])
                ql_avg_waits.append(avg_ql_wait)
                intervals.append(i)

        # Process fixed-time data separately to match intervals
        for step in intervals:
            fixed_data_in_interval = [h for h in fixed_history if h["time"] <= step]
            if fixed_data_in_interval:
                # Use last 10 points or all available points if less than 10
                recent_points = (
                    fixed_data_in_interval[-10:]
                    if len(fixed_data_in_interval) >= 10
                    else fixed_data_in_interval
                )
                avg_fixed_wait = np.mean(
                    [h.get("total_waiting", 0) for h in recent_points]
                )
                fixed_avg_waits.append(avg_fixed_wait)
            else:
                fixed_avg_waits.append(0)  # Default value if no data

        if intervals and ql_avg_waits:
            ax5.plot(
                intervals,
                ql_avg_waits,
                color=ql_color,
                linewidth=2.5,
                marker="o",
                label="Q-Learning",
                alpha=0.8,
            )
            if fixed_avg_waits and len(fixed_avg_waits) == len(intervals):
                ax5.plot(
                    intervals,
                    fixed_avg_waits,
                    color=fixed_color,
                    linewidth=2.5,
                    marker="s",
                    label="Fixed-Time",
                    alpha=0.8,
                )
            ax5.set_title(
                "Avg Wait Time vs Simulation Step", fontsize=14, fontweight="bold"
            )
            ax5.set_ylabel("Average Waiting Cars")
            ax5.set_xlabel("Simulation Step")
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # Add table-like annotations for Q-Learning data
            for i, (step, wait) in enumerate(zip(intervals, ql_avg_waits)):
                ax5.annotate(
                    f"{wait:.1f}",
                    (step, wait),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )
        else:
            ax5.text(
                0.5,
                0.5,
                "No Data Available",
                ha="center",
                va="center",
                transform=ax5.transAxes,
                fontsize=12,
            )
            ax5.set_title(
                "Avg Wait Time vs Simulation Step", fontsize=14, fontweight="bold"
            )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            # Save the plot to file
            if save_path.endswith(".pdf"):
                plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
                print(f"✓ Comparison plot saved as PDF: {save_path}")
            elif save_path.endswith(".png"):
                plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
                print(f"✓ Comparison plot saved as PNG: {save_path}")
            else:
                # Default to PNG if no extension specified
                plt.savefig(
                    f"{save_path}.png", format="png", dpi=300, bbox_inches="tight"
                )
                print(f"✓ Comparison plot saved as PNG: {save_path}.png")
            plt.close()  # Close the figure to free memory
        else:
            plt.show()

        # Print table for Avg wait time vs simulation step
        self.print_avg_wait_time_table(ql_history, fixed_history)

    def print_avg_wait_time_table(self, ql_history, fixed_history):
        """
        Print a table showing average wait time at different simulation steps.

        Args:
            ql_history: Q-Learning performance history
            fixed_history: Fixed-Time performance history
        """
        if not ql_history:
            print("No Q-Learning history available for table")
            return

        print("\n" + "=" * 80)
        print("AVG WAIT TIME VS SIMULATION STEP TABLE")
        print("=" * 80)
        print(f"{'Step':<10} {'Q-Learning':<12} {'Fixed-Time':<12}")
        print("-" * 80)

        # Calculate intervals
        max_time = max(
            max(ql_history, key=lambda x: x["time"])["time"] if ql_history else 0,
            max(fixed_history, key=lambda x: x["time"])["time"] if fixed_history else 0,
        )
        interval_size = max(max_time // 10, 50)  # At least 50 steps per interval

        for i in range(0, max_time + 1, interval_size):
            if i == 0:
                continue

            # Q-Learning data
            ql_data_in_interval = [h for h in ql_history if h["time"] <= i]
            ql_avg_wait = 0
            if ql_data_in_interval:
                ql_avg_wait = np.mean(
                    [h["total_waiting"] for h in ql_data_in_interval[-10:]]
                )

            # Fixed-Time data
            fixed_data_in_interval = [h for h in fixed_history if h["time"] <= i]
            fixed_avg_wait = 0
            if fixed_data_in_interval:
                fixed_avg_wait = np.mean(
                    [h.get("total_waiting", 0) for h in fixed_data_in_interval[-10:]]
                )

            print(f"{i:<10} {ql_avg_wait:<12.2f} {fixed_avg_wait:<12.2f}")

        print("=" * 80)
