"""
Example integration of Q-Learning mixin into reto4.py

This file shows how to integrate the QLearningTrafficMixin into the existing
TrafficModel class in reto4.py to enable adaptive traffic light control.
"""

# Add this import at the top of reto4.py
from traffic_qlearning_mixin import QLearningTrafficMixin

# Modify the TrafficModel class definition to inherit from the mixin
# Change this line in reto4.py:
# class TrafficModel(ap.Model):
#
# To this:
# class TrafficModel(ap.Model, QLearningTrafficMixin):

# Then add this to the setup() method in TrafficModel (after the existing setup code):
"""
    def setup(self):
        # ... existing setup code ...

        # Initialize Q-learning (add this)
        self.init_qlearning(
            alpha=0.1,        # Learning rate
            gamma=0.95,       # Discount factor
            epsilon=0.2,      # Initial exploration rate
            epsilon_decay=0.995,  # Exploration decay
            min_epsilon=0.05  # Minimum exploration rate
        )

        # ... rest of existing setup code ...
"""

# Modify the step() method in TrafficModel to include Q-learning decisions
# Add this to the step() method (replace the existing traffic light cycling logic):
"""
    def step(self):
        # ... existing step code ...

        # Q-learning traffic light control (replace the fixed cycling)
        if self.t % 5 == 0:  # Make decisions every 5 steps
            self.ql_step()

        # ... rest of existing step code (remove the old cycling logic) ...
"""

# Optional: Add Q-learning statistics to your analysis
# You can call this method to get Q-learning performance stats:
"""
    def get_qlearning_stats(self):
        ql_stats = self.get_ql_stats()
        base_stats = {
            'total_cars_spawned': self.total_cars_spawned,
            'total_cars_arrived': self.total_cars_arrived,
            'completion_rate': self.total_cars_arrived / max(self.total_cars_spawned, 1),
            'active_group': self.active_group
        }

        if ql_stats:
            base_stats.update({
                'ql_total_reward': ql_stats['total_reward'],
                'ql_q_table_size': ql_stats['q_table_size'],
                'ql_epsilon': ql_stats['epsilon']
            })

        return base_stats
"""

# Example of how to run a comparison between fixed and Q-learning control:
"""
def run_traffic_comparison():
    # Fixed time control
    print("Running Fixed-Time Control...")
    model_fixed = TrafficModel({
        "steps": 500,
        "spawn_rate": 1,
        "world_size": 500,
        "preset": "morning",
        "animation": False
    })
    model_fixed.enable_qlearning(False)  # Disable Q-learning
    model_fixed.run()

    # Q-learning control
    print("Running Q-Learning Control...")
    model_ql = TrafficModel({
        "steps": 500,
        "spawn_rate": 1,
        "world_size": 500,
        "preset": "morning",
        "animation": False
    })
    model_ql.enable_qlearning(True)  # Enable Q-learning
    model_ql.run()

    # Compare results
    fixed_stats = model_fixed.get_qlearning_stats()
    ql_stats = model_ql.get_qlearning_stats()

    print("
Comparison Results:")
    print(f"Fixed-Time - Completion Rate: {fixed_stats['completion_rate']:.1%}")
    print(f"Q-Learning - Completion Rate: {ql_stats['completion_rate']:.1%}")
    if 'ql_total_reward' in ql_stats:
        print(f"Q-Learning - Total Reward: {ql_stats['ql_total_reward']:.1f}")

    return model_fixed, model_ql
"""
