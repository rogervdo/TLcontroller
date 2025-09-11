"""
Q-Learning Mixin for Node-    def init_qlearning(
        self, alpha=0.4, gamma=0.95, epsilon=0.1, epsilon_decay=0.97, min_epsilon=0.02
    ):ed Traffic Simulation

This module provides a Q-learning controller mixin that can be integrated into
the existing TrafficModel to enable adaptive traffic light control based on
reinforcement learning.

The Q-learning agent learns to optimize traffic light group switching based on:
- Queue lengths at different traffic light groups
- Waiting times of vehicles
- Traffic flow efficiency
- Queue imbalance penalties

Adapted from the original 4-way intersection Q-learning controller to work with
the node-based traffic simulation with multiple traffic light groups.
"""

import numpy as np
from collections import defaultdict
import random


class QLearningTrafficMixin:
    """
    Q-Learning mixin for adaptive traffic light control in node-based simulations.

    This mixin adds Q-learning capabilities to the TrafficModel, allowing the
    traffic light system to learn optimal switching patterns based on traffic conditions.
    """

    def init_qlearning(
        self,
        alpha=0.25,
        gamma=0.95,
        epsilon=0.35,
        epsilon_decay=0.995,
        min_epsilon=0.05,
    ):
        """
        Initialize Q-learning parameters. Call this in the model's setup() method.

        Args:
            alpha: Learning rate (0.1)
            gamma: Discount factor (0.95)
            epsilon: Initial exploration rate (0.2)
            epsilon_decay: Rate at which epsilon decays (0.995)
            min_epsilon: Minimum exploration rate (0.05)
        """
        # Q-learning parameters
        self.ql_alpha = alpha
        self.ql_gamma = gamma
        self.ql_epsilon = epsilon
        self.ql_epsilon_decay = epsilon_decay
        self.ql_min_epsilon = min_epsilon

        # Q-table: state -> action -> Q-value
        self.Q = defaultdict(lambda: defaultdict(float))

        # Tracking variables
        self.last_state = None
        self.last_action = None
        self.total_reward = 0
        self.episode_rewards = []
        if not hasattr(self, "ql_enabled"):
            self.ql_enabled = True

        # Performance tracking
        self.ql_performance_history = []

        # Anti-stagnation mechanism
        self.steps_in_current_group = 0
        self.max_steps_per_group = (
            15  # Force switch after 15 steps (45 simulation steps)
        )

        print("Q-Learning traffic controller initialized")

    def get_traffic_state(self):
        """
        Convert current traffic conditions to a discrete state representation.

        Returns:
            tuple: State representation containing:
                - Queue lengths for each traffic group
                - Current active group
                - Time spent in current group
                - Total waiting cars
        """
        # Count waiting cars by traffic light group with enhanced approaching car detection
        group_queues = {0: 0, 1: 0, 2: 0}
        group_approaching = {0: 0, 1: 0, 2: 0}  # Cars approaching but not yet at light
        total_waiting = 0
        waiting_time_distribution = {0: 0, 1: 0, 2: 0, 3: 0}  # 0-3s, 4-7s, 8-15s, >15s

        # Identify nodes that are controlled by traffic lights
        stoplight_nodes = ["8_16", "8_14", "8_12", "18_16", "18_14", "18_12"]

        for car in self.cars:
            if car.state == "moving":
                continue  # Only count stopped/waiting cars

            # Check if car is waiting at or approaching a traffic light
            current_node = self.road_network.nodes.get(car.current_node_id)
            if current_node and current_node.group_id is not None:
                # Car is at a traffic light node
                if car.current_node_id in stoplight_nodes:
                    # Check if this group's light is red
                    if current_node.group_id != self.active_group:
                        group_queues[current_node.group_id] += 1
                        total_waiting += 1

                        # Categorize waiting time for state representation
                        if hasattr(car, "wait_steps"):
                            wait_time = car.wait_steps
                            if wait_time <= 3:
                                waiting_time_distribution[0] += 1
                            elif wait_time <= 7:
                                waiting_time_distribution[1] += 1
                            elif wait_time <= 15:
                                waiting_time_distribution[2] += 1
                            else:
                                waiting_time_distribution[3] += 1
                        else:
                            waiting_time_distribution[0] += 1
                else:
                    # Car is approaching a traffic light - check multiple steps ahead
                    if car.path and car.current_path_index < len(car.path) - 1:
                        # Check next 2 nodes in path for traffic lights
                        for steps_ahead in range(
                            1, min(3, len(car.path) - car.current_path_index)
                        ):
                            next_node_id = car.path[
                                car.current_path_index + steps_ahead
                            ]
                            next_node = self.road_network.nodes.get(next_node_id)
                            if (
                                next_node
                                and next_node.group_id is not None
                                and next_node.group_id != self.active_group
                            ):
                                group_approaching[next_node.group_id] += 1
                                total_waiting += 1
                                waiting_time_distribution[0] += (
                                    1  # Approaching cars have minimal wait time
                                )
                                break  # Only count once per car

        # Discretize queue lengths
        def discretize_queue(count):
            if count == 0:
                return 0
            elif count <= 2:
                return 1
            elif count <= 5:
                return 2
            elif count <= 10:
                return 3
            else:
                return 4  # Very long queue

        # Time spent in current group (discretized)
        time_in_group = min(self.t % (self.group_cycle_steps * 3), 15) // 3

        # Create enhanced state tuple with waiting time distribution and approaching cars
        state = (
            discretize_queue(group_queues[0]),  # Group 0 queue
            discretize_queue(group_queues[1]),  # Group 1 queue
            discretize_queue(group_queues[2]),  # Group 2 queue
            discretize_queue(group_approaching[0]),  # Group 0 approaching
            discretize_queue(group_approaching[1]),  # Group 1 approaching
            discretize_queue(group_approaching[2]),  # Group 2 approaching
            self.active_group,  # Current active group
            time_in_group,  # Time in current group
            discretize_queue(total_waiting),  # Total waiting cars
            waiting_time_distribution[0],  # Cars waiting 0-3 steps
            waiting_time_distribution[1],  # Cars waiting 4-7 steps
            waiting_time_distribution[2],  # Cars waiting 8-15 steps
            waiting_time_distribution[3],  # Cars waiting >15 steps
        )

        return state

    def get_traffic_actions(self):
        """
        Get available actions for the traffic light controller.

        Returns:
            list: Available actions [0=stay in current group, 1=switch to next group]
        """
        return [0, 1]  # 0: maintain current group, 1: switch to next

    def choose_traffic_action(self, state):
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current state representation

        Returns:
            int: Chosen action (0 or 1)
        """
        if not self.ql_enabled:
            return 0  # Default to maintaining current group

        if random.random() < self.ql_epsilon:
            # Exploration: random action
            return random.choice(self.get_traffic_actions())
        else:
            # Exploitation: best action according to Q-table
            actions = self.get_traffic_actions()
            q_values = [self.Q[state][action] for action in actions]
            return actions[np.argmax(q_values)]

    def calculate_traffic_reward(self):
        """
        Calculate reward based on current traffic conditions.
        Enhanced version with better waiting car contribution inspired by cruceqlearningdump.

        Returns:
            tuple: (reward, metrics_dict) where metrics_dict contains detailed waiting car statistics
        """
        # Count waiting cars by group with detailed waiting time analysis
        group_waiting = {0: 0, 1: 0, 2: 0}
        total_waiting = 0
        long_waiting = 0  # Cars waiting more than 10 steps
        early_long_waiting = 0  # Cars waiting more than 7 steps (early starvation)
        very_long_waiting = 0  # Cars waiting more than 15 steps (severe starvation)

        stoplight_nodes = ["8_16", "8_14", "8_12", "18_16", "18_14", "18_12"]

        # Track cars that are actively moving through intersection during green phase
        cars_served_this_step = 0

        for car in self.cars:
            if car.state == "moving":
                # Check if car just moved through an intersection during green phase
                current_node = self.road_network.nodes.get(car.current_node_id)
                if (
                    current_node
                    and current_node.group_id is not None
                    and current_node.group_id == self.active_group
                ):
                    cars_served_this_step += 1
                continue

            # Check if car is affected by current traffic light state
            current_node = self.road_network.nodes.get(car.current_node_id)
            if current_node and current_node.group_id is not None:
                if car.current_node_id in stoplight_nodes:
                    if current_node.group_id != self.active_group:
                        group_waiting[current_node.group_id] += 1
                        total_waiting += 1

                        # Enhanced waiting time tracking
                        if hasattr(car, "wait_steps"):
                            car.wait_steps += 1
                            wait_time = car.wait_steps

                            # Categorize waiting times for different penalty levels
                            if wait_time > 15:
                                very_long_waiting += 1
                            elif wait_time > 10:
                                long_waiting += 1
                            elif wait_time > 7:
                                early_long_waiting += 1
                        else:
                            car.wait_steps = 1
                    else:
                        # Car can move, reset wait counter
                        if hasattr(car, "wait_steps"):
                            car.wait_steps = 0
                else:
                    # Check approaching cars
                    if car.path and car.current_path_index < len(car.path) - 1:
                        next_node_id = car.path[car.current_path_index + 1]
                        next_node = self.road_network.nodes.get(next_node_id)
                        if (
                            next_node
                            and next_node.group_id is not None
                            and next_node.group_id != self.active_group
                        ):
                            group_waiting[next_node.group_id] += 1
                            total_waiting += 1

        # Enhanced reward components with LINEAR penalties instead of threshold-based

        # 1. Linear penalty for waiting cars - applied incrementally per step
        # Base penalty: -0.5 per waiting car per step
        base_waiting_penalty = -total_waiting * 0.5

        # 2. Additional linear penalty for cars waiting longer
        # For each car waiting >7 steps: additional -0.3 per step beyond 7
        extended_waiting_penalty = 0
        # For each car waiting >10 steps: additional -0.8 per step beyond 10
        long_waiting_penalty = 0
        # For each car waiting >15 steps: additional -2.0 per step beyond 15
        very_long_waiting_penalty = 0

        # Calculate linear penalties for each waiting car
        for car in self.cars:
            if car.state != "moving" and hasattr(car, "wait_steps"):
                wait_time = car.wait_steps
                if wait_time > 0:
                    # Base penalty already included in base_waiting_penalty

                    # Additional penalty for extended waiting (>7 steps)
                    if wait_time > 7:
                        extended_steps = wait_time - 7
                        extended_waiting_penalty -= extended_steps * 0.3

                    # Additional penalty for long waiting (>10 steps)
                    if wait_time > 10:
                        long_steps = wait_time - 10
                        long_waiting_penalty -= long_steps * 0.8

                    # Additional penalty for very long waiting (>15 steps)
                    if wait_time > 15:
                        very_long_steps = wait_time - 15
                        very_long_waiting_penalty -= very_long_steps * 2.0

        # 3. Penalty for queue imbalance - MAINTAIN FAIRNESS
        waiting_counts = list(group_waiting.values())
        if waiting_counts and sum(waiting_counts) > 0:
            avg_waiting = np.mean(waiting_counts)
            imbalance_penalty = (
                -np.sum([abs(w - avg_waiting) for w in waiting_counts])
                * 2.0  # Maintained
            )
        else:
            imbalance_penalty = 0

        # 4. Small reward for immediate throughput - MINIMAL IMPORTANCE
        throughput_reward = cars_served_this_step * 0.01  # Further decreased

        # 5. Tiny reward for overall completion - MINIMAL IMPORTANCE
        completion_reward = self.total_cars_arrived * 0.2  # Further decreased

        # 6. Strong penalty for staying too long in same group - PREVENT STUCK STATES
        time_in_group = self.t % (self.group_cycle_steps * 3)
        if time_in_group > self.group_cycle_steps * 2.5:  # 2.5x normal cycle time
            time_penalty = -15.0  # Increased from -8.0
        elif time_in_group > self.group_cycle_steps * 2:  # 2x normal cycle time
            time_penalty = -8.0  # Increased from -5.0
        else:
            time_penalty = 0.0

        # 7. Small reward for maintaining balance when queues are low
        balance_bonus = 0
        if total_waiting <= 2 and max(waiting_counts) - min(waiting_counts) <= 1:
            balance_bonus = 0.5

        # Combine all reward components with linear penalties
        reward = (
            base_waiting_penalty
            + extended_waiting_penalty
            + long_waiting_penalty
            + very_long_waiting_penalty
            + imbalance_penalty
            + throughput_reward
            + completion_reward
            + time_penalty
            + balance_bonus
        )

        # Return both reward and detailed metrics
        metrics = {
            "total_waiting": total_waiting,
            "long_waiting": long_waiting,
            "early_long_waiting": early_long_waiting,
            "very_long_waiting": very_long_waiting,
            "group_waiting": group_waiting.copy(),
            "cars_served_this_step": cars_served_this_step,
            "time_in_group": time_in_group,
            "queue_imbalance": abs(
                max(group_waiting.values()) - min(group_waiting.values())
            )
            if group_waiting
            else 0,
            "reward_components": {
                "base_waiting_penalty": base_waiting_penalty,
                "extended_waiting_penalty": extended_waiting_penalty,
                "long_waiting_penalty": long_waiting_penalty,
                "very_long_waiting_penalty": very_long_waiting_penalty,
                "imbalance_penalty": imbalance_penalty,
                "throughput_reward": throughput_reward,
                "completion_reward": completion_reward,
                "time_penalty": time_penalty,
                "balance_bonus": balance_bonus,
            },
        }

        return reward, metrics

    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        if not self.ql_enabled:
            return

        # Q-learning update
        current_q = self.Q[state][action]
        if next_state is not None:
            next_actions = self.get_traffic_actions()
            next_q_values = [self.Q[next_state][a] for a in next_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
            new_q = current_q + self.ql_alpha * (
                reward + self.ql_gamma * max_next_q - current_q
            )
        else:
            new_q = current_q + self.ql_alpha * (reward - current_q)

        self.Q[state][action] = new_q

    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.ql_enabled:
            self.ql_epsilon = max(
                self.ql_min_epsilon, self.ql_epsilon * self.ql_epsilon_decay
            )

    def ql_step(self):
        """
        Perform Q-learning step. Call this in the model's step() method.

        This method should be called at regular intervals (e.g., every few steps)
        to allow the Q-learning agent to observe the traffic state and make decisions.
        """
        if not self.ql_enabled:
            return

        # Get current state
        current_state = self.get_traffic_state()

        # Calculate reward from last action
        if self.last_state is not None and self.last_action is not None:
            reward, metrics = self.calculate_traffic_reward()
            self.total_reward += reward

            # Update Q-value
            self.update_q_value(
                self.last_state, self.last_action, reward, current_state
            )

            # Track performance with enhanced waiting car metrics
            self.ql_performance_history.append(
                {
                    "time": self.t,
                    "state": self.last_state,
                    "action": self.last_action,
                    "reward": reward,
                    "epsilon": self.ql_epsilon,  # Add epsilon tracking
                    "total_waiting": metrics["total_waiting"],
                    "long_waiting": metrics["long_waiting"],
                    "early_long_waiting": metrics["early_long_waiting"],
                    "very_long_waiting": metrics["very_long_waiting"],
                    "group_waiting": metrics["group_waiting"],
                    "cars_served_this_step": metrics["cars_served_this_step"],
                    "active_group": self.active_group,
                    "time_in_group": metrics["time_in_group"],
                    "queue_imbalance": metrics["queue_imbalance"],
                    "reward_components": metrics["reward_components"],
                }
            )

        # Choose action
        action = self.choose_traffic_action(current_state)

        # Anti-stagnation mechanism: force switch if too long in same group
        if self.steps_in_current_group >= self.max_steps_per_group:
            action = 1  # Force switch
            print(f"Q-Learning: Forced switch due to stagnation (t={self.t})")

        # Execute action
        if action == 1:  # Switch to next group
            old_group = self.active_group
            self.active_group = (self.active_group + 1) % 3
            self.steps_in_current_group = 0  # Reset counter
            print(
                f"Q-Learning: Switched from Group {old_group} to Group {self.active_group} (t={self.t})"
            )
        else:
            # Action 0: Stay in current group
            self.steps_in_current_group += 1
            print(
                f"Q-Learning: Maintaining Group {self.active_group} (t={self.t}, steps={self.steps_in_current_group})"
            )

        # Store for next iteration
        self.last_state = current_state
        self.last_action = action

        # Decay epsilon
        self.decay_epsilon()

    def get_ql_stats(self):
        """
        Get Q-learning statistics including enhanced waiting car metrics.

        Returns:
            dict: Statistics about Q-learning performance with detailed reward breakdown
        """
        if not self.ql_enabled:
            return {}

        # Calculate current reward components for analysis
        if self.ql_performance_history:
            latest_metrics = self.ql_performance_history[-1]
            reward_components = latest_metrics.get("reward_components", {})

            # Calculate averages from recent performance history
            recent_history = self.ql_performance_history[-50:]  # Last 50 entries
            avg_waiting_cars = np.mean(
                [h.get("total_waiting", 0) for h in recent_history]
            )
            avg_long_waiting = np.mean(
                [h.get("long_waiting", 0) for h in recent_history]
            )
            avg_early_long_waiting = np.mean(
                [h.get("early_long_waiting", 0) for h in recent_history]
            )
            avg_very_long_waiting = np.mean(
                [h.get("very_long_waiting", 0) for h in recent_history]
            )
            avg_cars_served = np.mean(
                [h.get("cars_served_this_step", 0) for h in recent_history]
            )
        else:
            reward_components = {}
            avg_waiting_cars = avg_long_waiting = avg_early_long_waiting = (
                avg_very_long_waiting
            ) = avg_cars_served = 0

        return {
            "total_reward": self.total_reward,
            "q_table_size": len(self.Q),
            "epsilon": self.ql_epsilon,
            "performance_history": self.ql_performance_history[-100:]
            if self.ql_performance_history
            else [],
            # Enhanced waiting car metrics
            "waiting_car_analysis": {
                "avg_total_waiting": avg_waiting_cars,
                "avg_long_waiting": avg_long_waiting,  # >10 steps
                "avg_early_long_waiting": avg_early_long_waiting,  # >7 steps
                "avg_very_long_waiting": avg_very_long_waiting,  # >15 steps
                "avg_cars_served_per_step": avg_cars_served,
            },
            # Current reward component breakdown
            "current_reward_breakdown": reward_components,
            # Reward contribution percentages (if available)
            "reward_contribution_analysis": self._analyze_reward_contributions(
                reward_components
            )
            if reward_components
            else {},
        }

    def _analyze_reward_contributions(self, reward_components):
        """
        Analyze the contribution of different reward components.

        Args:
            reward_components: Dictionary of reward component values

        Returns:
            dict: Analysis of reward contributions
        """
        if not reward_components:
            return {}

        total_reward = sum(reward_components.values())

        if total_reward == 0:
            return {k: 0.0 for k in reward_components.keys()}

        # Calculate percentage contributions
        contributions = {}
        for component, value in reward_components.items():
            contributions[f"{component}_pct"] = (value / total_reward) * 100

        # Identify dominant components
        sorted_components = sorted(
            reward_components.items(), key=lambda x: x[1], reverse=True
        )
        dominant_positive = [k for k, v in sorted_components if v > 0][:3]
        dominant_negative = [k for k, v in sorted_components if v < 0][:3]

        return {
            "total_reward": total_reward,
            "component_contributions": contributions,
            "dominant_positive_components": dominant_positive,
            "dominant_negative_components": dominant_negative,
            "waiting_car_penalty_intensity": self._calculate_waiting_penalty_intensity(
                reward_components
            ),
        }

    def _calculate_waiting_penalty_intensity(self, reward_components):
        """
        Calculate the intensity of waiting car penalties.

        Args:
            reward_components: Dictionary of reward component values

        Returns:
            str: Description of waiting penalty intensity
        """
        waiting_penalties = [
            reward_components.get("base_waiting_penalty", 0),
            reward_components.get("extended_waiting_penalty", 0),
            reward_components.get("long_waiting_penalty", 0),
            reward_components.get("very_long_waiting_penalty", 0),
            reward_components.get("imbalance_penalty", 0),
        ]

        total_waiting_penalty = sum(waiting_penalties)

        if total_waiting_penalty >= -1.0:
            return "low"
        elif total_waiting_penalty >= -5.0:
            return "moderate"
        elif total_waiting_penalty >= -15.0:
            return "high"
        else:
            return "critical"


# Example usage in TrafficModel:
"""
class TrafficModel(ap.Model, QLearningTrafficMixin):
    def setup(self):
        # ... existing setup code ...

        # Initialize Q-learning
        self.__init_qlearning(
            alpha=0.1,
            gamma=0.95,
            epsilon=0.2,
            epsilon_decay=0.995,
            min_epsilon=0.05
        )

    def step(self):
        # ... existing step code ...

        # Q-learning decision every 5 steps
        if self.t % 5 == 0:
            self.ql_step()

        # ... rest of step code ...
"""
