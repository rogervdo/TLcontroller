"""
Q-Learning Mixin for Node-Based Traffic Simulation

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
        self, alpha=0.1, gamma=0.95, epsilon=0.2, epsilon_decay=0.995, min_epsilon=0.05
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
        # Count waiting cars by traffic light group
        group_queues = {0: 0, 1: 0, 2: 0}
        total_waiting = 0

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
                else:
                    # Car is approaching a traffic light
                    # Check next nodes in path to see if they lead to a red light
                    if car.path and car.current_path_index < len(car.path) - 1:
                        next_node_id = car.path[car.current_path_index + 1]
                        next_node = self.road_network.nodes.get(next_node_id)
                        if (
                            next_node
                            and next_node.group_id is not None
                            and next_node.group_id != self.active_group
                        ):
                            group_queues[next_node.group_id] += 1
                            total_waiting += 1

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

        # Create state tuple
        state = (
            discretize_queue(group_queues[0]),  # Group 0 queue
            discretize_queue(group_queues[1]),  # Group 1 queue
            discretize_queue(group_queues[2]),  # Group 2 queue
            self.active_group,  # Current active group
            time_in_group,  # Time in current group
            discretize_queue(total_waiting),  # Total waiting cars
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

        Returns:
            float: Reward value
        """
        # Count waiting cars by group
        group_waiting = {0: 0, 1: 0, 2: 0}
        total_waiting = 0
        long_waiting = 0  # Cars waiting more than 10 steps

        stoplight_nodes = ["8_16", "8_14", "8_12", "18_16", "18_14", "18_12"]

        for car in self.cars:
            if car.state == "moving":
                continue

            # Check if car is affected by current traffic light state
            current_node = self.road_network.nodes.get(car.current_node_id)
            if current_node and current_node.group_id is not None:
                if car.current_node_id in stoplight_nodes:
                    if current_node.group_id != self.active_group:
                        group_waiting[current_node.group_id] += 1
                        total_waiting += 1
                        # Count cars waiting too long (simulate wait_time)
                        if hasattr(car, "wait_steps"):
                            car.wait_steps += 1
                            if car.wait_steps > 10:
                                long_waiting += 1
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

        # Reward components
        # 1. Base penalty for total waiting cars (encourage flow)
        queue_penalty = -total_waiting * 0.5

        # 2. Penalty for cars waiting too long (starvation)
        starvation_penalty = -long_waiting * 2.0

        # 3. Penalty for queue imbalance (fairness)
        waiting_counts = list(group_waiting.values())
        if waiting_counts:
            avg_waiting = np.mean(waiting_counts)
            imbalance_penalty = (
                -np.sum([abs(w - avg_waiting) for w in waiting_counts]) * 0.3
            )
        else:
            imbalance_penalty = 0

        # 4. Reward for cars that completed journeys (throughput)
        throughput_reward = self.total_cars_arrived * 0.1

        # 5. Penalty for staying too long in same group
        time_in_group = self.t % (self.group_cycle_steps * 3)
        if time_in_group > self.group_cycle_steps * 2:
            time_penalty = -1.0
        else:
            time_penalty = 0.0

        # Combine rewards
        reward = (
            queue_penalty
            + starvation_penalty
            + imbalance_penalty
            + throughput_reward
            + time_penalty
        )

        return reward

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
            reward = self.calculate_traffic_reward()
            self.total_reward += reward

            # Update Q-value
            self.update_q_value(
                self.last_state, self.last_action, reward, current_state
            )

            # Track performance
            self.ql_performance_history.append(
                {
                    "time": self.t,
                    "state": self.last_state,
                    "action": self.last_action,
                    "reward": reward,
                    "total_waiting": sum(
                        [1 for car in self.cars if car.state != "moving"]
                    ),
                    "active_group": self.active_group,
                }
            )

        # Choose action
        action = self.choose_traffic_action(current_state)

        # Execute action
        if action == 1:  # Switch to next group
            self.active_group = (self.active_group + 1) % 3
            print(f"Q-Learning: Switched to Group {self.active_group} (t={self.t})")

        # Store for next iteration
        self.last_state = current_state
        self.last_action = action

        # Decay epsilon
        self.decay_epsilon()

    def get_ql_stats(self):
        """
        Get Q-learning statistics.

        Returns:
            dict: Statistics about Q-learning performance
        """
        if not self.ql_enabled:
            return {}

        return {
            "total_reward": self.total_reward,
            "q_table_size": len(self.Q),
            "epsilon": self.ql_epsilon,
            "performance_history": self.ql_performance_history[-100:]
            if self.ql_performance_history
            else [],
        }

    def enable_qlearning(self, enabled=True):
        """
        Enable or disable Q-learning.

        Args:
            enabled: Whether to enable Q-learning
        """
        # Initialize Q-learning if not already done
        if not hasattr(self, "ql_enabled"):
            self.init_qlearning()

        self.ql_enabled = enabled
        if enabled:
            print("Q-Learning enabled")
        else:
            print("Q-Learning disabled")


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
