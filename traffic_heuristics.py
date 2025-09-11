"""
Traffic Heuristics Mixin

This module provides a mixin class for adaptive traffic light control based on car counts.
The heuristics monitor traffic congestion in different groups and adjust light timing accordingly.

Features:
- Car counting for each traffic light group (expanded node coverage)
- Density-based traffic analysis (cars per node)
- Adaptive timer adjustment based on comparative traffic density
- Automatic reduction of green light duration when other groups have higher congestion

Usage:
    from traffic_heuristics import HeuristicsMixin

    class TrafficModel(ap.Model, HeuristicsMixin):
        def setup(self):
            # ... existing setup code ...
            self.group_nodes = {
                0: ["8_18", "8_20", "8_22", "8_24", "18_10", "18_8", "18_6", "18_4"],
                1: ["6_14", "4_14", "2_14", "0_14", "6_12", "4_12", "2_12", "0_12", "16_13", "14_12", "12_12", "10_13", "16_11", "14_10", "12_10", "10_11"],
                2: ["20_15", "22_15", "24_15", "26_15", "10_15", "12_16", "14_16", "16_15", "10_17", "12_18", "14_18", "16_17"],
            }
            # ... rest of setup ...

        def step(self):
            # ... existing step code ...
            if self.t % self.group_cycle_steps == 0 and self.t > 0:
                self.active_group = (self.active_group + 1) % 3
                if hasattr(self, 'adjust_timer_based_on_traffic'):
                    self.adjust_timer_based_on_traffic()
            # ... rest of step ...
"""


class HeuristicsMixin:
    """Mixin to add heuristics-based traffic light control"""

    def get_group_car_count(self, group):
        """Count cars in the nodes associated with a traffic light group"""
        count = 0
        for node_id in self.group_nodes[group]:
            if self.node_occupancy[node_id] is not None:
                count += 1
        return count

    def get_group_density(self, group):
        """Get car density (cars per node) for a group"""
        node_count = len(self.group_nodes[group])
        if node_count == 0:
            return 0
        car_count = self.get_group_car_count(group)
        return car_count / node_count

    def adjust_timer_based_on_traffic(self):
        """Adjust traffic light timer based on car density in different groups"""
        current_density = self.get_group_density(self.active_group)
        other_densities = [
            self.get_group_density(g) for g in range(3) if g != self.active_group
        ]

        # Use relative threshold based on group size
        max_other_density = max(other_densities) if other_densities else 0

        if max_other_density > current_density:
            # Reduce timer if other groups have higher density
            old_timer = self.group_cycle_steps
            self.group_cycle_steps = max(1, self.group_cycle_steps - 1)
            if old_timer != self.group_cycle_steps:
                print(".2f")
        else:
            # Reset to base timer if current group has priority
            old_timer = self.group_cycle_steps
            self.group_cycle_steps = 5
            if old_timer != self.group_cycle_steps:
                print(f"Reset timer to {self.group_cycle_steps}")

    def get_traffic_stats(self):
        """Get current traffic statistics for all groups"""
        stats = {}
        for group in range(3):
            stats[f"group_{group}"] = {
                "car_count": self.get_group_car_count(group),
                "node_count": len(self.group_nodes[group]),
                "density": self.get_group_density(group),
                "nodes": self.group_nodes[group],
            }
        stats["active_group"] = self.active_group
        stats["current_timer"] = self.group_cycle_steps
        return stats
