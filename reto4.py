import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import heapq
import socket
import json

"""
Traffic Simulation with Unity Integration

This script provides a comprehensive traffic simulation with the ability to send data to Unity for visualization.

Features:
- Node-based road network with A* pathfinding
- Traffic lights with cycling groups
- Yield behaviors at merge points
- Weighted destination spawning
- Real-time JSON data export for Unity integration

Unity Integration:
- Use run_simulation_and_send_json() to run simulation and send complete data to Unity
- Use send_current_simulation_state_to_unity(model) for real-time state updates
- Use run_simulation_and_save_json() to save data without sending to Unity
- Data includes: car positions, states, spawn/target nodes, traffic light groups, and network topology

JSON Data Structure:
{
    "time": simulation_step,
    "active_group": current_traffic_light_group,
    "cars": [
        {
            "id": car_index,
            "x": position_x,
            "y": position_y,
            "state": "moving"|"arrived",
            "spawn_node": starting_node_id,
            "target_node": destination_node_id
        }
    ]
}
"""

# Simulation parameters

# Simulation parameters
params = {
    "steps": 100,
    "spawn_rate": 1,  # Probability of spawning a car each step
    "world_size": 500,  # World boundaries (0 to world_size)
    "preset": "morning",  # Traffic preset: "morning", "evening", "night"
    "animation": True,  # Save to gif
}


class TrafficNode:
    """Represents a node in the road network"""

    def __init__(self, node_id, x, y, node_type="intersection", group_id=None):
        self.id = node_id
        self.x = x
        self.y = y
        self.node_type = node_type  # "spawn", "intersection", "destination"
        self.group_id = (
            group_id  # Group ID for stoplight functionality (None if not applicable)
        )
        self.connected_nodes = {}  # {node_id: distance}

    def add_connection(self, target_node_id, distance):
        """Add a connection to another node"""
        self.connected_nodes[target_node_id] = distance

    def get_position(self):
        return np.array([self.x, self.y])


class Car(ap.Agent):
    """Car agent that moves between nodes"""

    def setup(self, start_node_id, target_node_id):
        self.current_node_id = start_node_id
        self.target_node_id = target_node_id
        self.path = []  # Will store the path as list of node IDs
        self.current_path_index = 0
        self.state = "moving"  # 'moving', 'arrived'

        # Assign color based on spawn node
        spawn_colors = {
            "8_26": "#4169E1",  # Royal blue
            "18_2": "#32CD32",  # Lime green
            "2_14": "#FF9500",  # Deep pink
            "2_12": "#FFD700",  # Gold
            "22_4": "#9932CC",  # Dark orchid
            "30_15": "#FF4500",  # Orange red
        }
        self.color = spawn_colors.get(start_node_id, "white")

        # Calculate initial path
        self.calculate_path()

        # Set initial position
        if self.path:
            start_node = self.model.road_network.nodes[self.current_node_id]
            self.position = start_node.get_position().copy()
            # Set occupancy
            self.model.node_occupancy[self.current_node_id] = self.id
        else:
            self.state = "arrived"

    def calculate_path(self):
        """Pathfinding using A* algorithm for optimal paths"""
        network = self.model.road_network

        # If direct connection exists, use it
        if self.target_node_id in network.nodes[self.current_node_id].connected_nodes:
            self.path = [self.current_node_id, self.target_node_id]
        else:
            # Use A* pathfinding for optimal path
            self.path = self.find_astar_path()

    def find_astar_path(self):
        """Find optimal path using A* algorithm"""
        network = self.model.road_network
        start = self.current_node_id
        goal = self.target_node_id

        # Priority queue for open set: (f_score, node_id)
        open_set = []
        heapq.heappush(open_set, (0, start))

        # Came from dictionary to reconstruct path
        came_from = {}

        # g_score: cost from start to current node
        g_score = {node_id: float("inf") for node_id in network.nodes}
        g_score[start] = 0

        # f_score: estimated total cost from start to goal through current node
        f_score = {node_id: float("inf") for node_id in network.nodes}
        f_score[start] = self.heuristic(start, goal)

        while open_set:
            # Get node with lowest f_score
            current_f, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                return self.reconstruct_path(came_from, current)

            # Check all neighbors
            for neighbor in network.nodes[current].connected_nodes:
                # Calculate tentative g_score
                edge_distance = network.nodes[current].connected_nodes[neighbor]
                tentative_g_score = g_score[current] + edge_distance

                if tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(
                        neighbor, goal
                    )

                    # Add to open set if not already there
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return []

    def heuristic(self, node_id1, node_id2):
        """Euclidean distance heuristic"""
        network = self.model.road_network
        pos1 = network.nodes[node_id1].get_position()
        pos2 = network.nodes[node_id2].get_position()
        return np.linalg.norm(pos2 - pos1)

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def step(self):
        if self.state == "arrived" or not self.path:
            return

        # Check if we've reached the end of our path
        if self.current_path_index >= len(self.path) - 1:
            self.state = "arrived"
            # Immediately clear occupancy when arriving at destination
            self.model.node_occupancy[self.current_node_id] = None
            return

        # Get next node
        next_node_id = self.path[self.current_path_index + 1]
        next_node = self.model.road_network.nodes[next_node_id]

        # Check if next node is occupied
        if self.model.node_occupancy[next_node_id] is not None:
            return  # Wait for the node to be free

        # Yield check for merge points
        for rule in self.model.yield_rules:
            if self.current_node_id == rule["yield"] and next_node_id == rule["merge"]:
                if self.model.node_occupancy[rule["priority"]] is not None:
                    return  # Yield to priority car

        # Group check for stoplight functionality
        current_node = self.model.road_network.nodes[self.current_node_id]
        if current_node.group_id is not None:
            stoplight_nodes = [
                "8_16",
                "8_14",
                "8_12",
                "18_16",
                "18_14",
                "18_12",
            ]  # Example stoplight nodes
            if next_node_id in stoplight_nodes:
                if current_node.group_id != self.model.active_group:
                    return  # Wait for green light

        # Move to next node instantly
        self.position = next_node.get_position().copy()
        # Update occupancy
        self.model.node_occupancy[self.current_node_id] = None
        self.model.node_occupancy[next_node_id] = self.id
        self.current_node_id = next_node_id
        self.current_path_index += 1


class RoadNetwork:
    """Manages the road network of nodes and connections"""

    def __init__(self):
        self.nodes = {}
        self.spawn_nodes = []
        self.destination_nodes = []

    def add_node(self, node_id, x, y, node_type="intersection", group_id=None):
        """Add a node to the network"""
        node = TrafficNode(node_id, x, y, node_type, group_id)
        self.nodes[node_id] = node

        if node_type == "spawn":
            self.spawn_nodes.append(node_id)
        elif node_type == "destination":
            self.destination_nodes.append(node_id)

    def add_edge(self, node_id1, node_id2):
        """Add bidirectional connection between two nodes"""
        if node_id1 in self.nodes and node_id2 in self.nodes:
            pos1 = self.nodes[node_id1].get_position()
            pos2 = self.nodes[node_id2].get_position()
            distance = np.linalg.norm(pos2 - pos1)

            self.nodes[node_id1].add_connection(node_id2, distance)
            self.nodes[node_id2].add_connection(node_id1, distance)

    def create_simple_network(self, world_size):
        # Additional new path nodes (user request)

        # New path nodes (user request)
        """Create custom road network based on specified layout"""
        # Manual positions for nodes, independent of x,y calculations
        positions = {
            "8_2": (49, -99),
            "8_4": (48, -64.7),
            "8_6": (47, -28.4),
            "8_8": (47, 12),
            "8_10": (47, 51),
            "8_12": (47, 83),
            "8_14": (48, 117),
            "8_16": (49, 175),
            "8_18": (50, 200),
            "8_20": (49.6, 244),
            "8_22": (49, 303),
            "8_24": (49, 350),
            "8_26": (48, 397),
            "18_2": (176.6, -95.8),
            "18_4": (175, -11),
            "18_6": (172.3, -57),
            "18_8": (173, -3),
            "18_10": (172, 36.9),
            "18_12": (171, 57),
            "18_14": (170.8, 74),
            "18_16": (173, 150),
            "18_18": (172, 196),
            "18_20": (172.7, 244),
            "18_22": (170, 307),
            "7_19": (28, 226.6),
            "5_17": (9.5, 209),
            "4_16": (-20, 190),
            "-2_14": (-160, 195.7),
            "0_14": (-122, 184),
            "2_14": (-78, 170),
            "4_14": (-38, 157.7),
            "6_14": (12.7, 138.5),
            "-2_12": (-173, 162),
            "0_12": (-126, 150.9),
            "2_12": (-82.5, 138),
            "4_12": (-40.7, 126.4),
            "6_12": (-3, 112.9),
            "30_15": (408, 55),
            "28_15": (361, 72),
            "26_15": (327, 85),
            "24_15": (295, 99),
            "22_15": (261, 118),
            "20_15": (223, 135),
            "10_13": (72.3, 87),
            "12_12": (94.2, 78),
            "14_12": (126, 71.9),
            "16_13": (150, 73.1),
            "6_16": (9.2, 181.7),
            "2_16": (-65, 203),
            "20_12": (214, 52.7),
            "22_12": (256, 36.6),
            "24_12": (285.6, 23.5),
            "26_12": (328, 8.9),
            "18_24": (169, 364),
            "18_26": (168, 403),
            "10_11": (70, 67),
            "12_10": (95, 56),
            "14_10": (125, 49.9),
            "16_11": (151, 51),
            "16_17": (159, 175),
            "14_18": (136, 187),
            "12_18": (108, 194),
            "10_17": (81, 189),
            "16_7": (153.7, -28.3),
            "14_8": (130, -16),
            "12_8": (97.7, -18),
            "10_7": (72.3, -34),
            "10_21": (76.5, 269),
            "12_20": (102, 260),
            "14_20": (137, 266),
            "16_21": (157, 276),
            "16_15": (160, 155),
            "14_16": (140, 165),
            "12_16": (107, 173),
            "10_15": (75, 169),
            "6_9": (-9, 66),
            "6_7": (-4.7, 28.9),
            "6_5": (-0.5, -12.1),
            "6_3": (2.9, -80),
            "22_4": (239, -110),
            "22_6": (240, -73),
            "22_8": (241, -12),
            "22_10": (249, 19),
        }
        nodes_list = [
            ("8_2", "destination", None),
            ("8_4", "intersection", None),
            ("8_6", "intersection", None),
            ("8_8", "intersection", None),
            ("8_10", "intersection", None),
            ("8_12", "intersection", None),
            ("8_14", "intersection", None),
            ("8_16", "intersection", None),
            ("8_18", "intersection", 0),
            ("8_20", "intersection", None),
            ("8_22", "intersection", None),
            ("8_24", "intersection", None),
            ("8_26", "spawn", None),
            ("18_2", "spawn", None),
            ("18_4", "intersection", None),
            ("18_6", "intersection", None),
            ("18_8", "intersection", None),
            ("18_10", "intersection", 0),
            ("18_12", "intersection", None),
            ("18_14", "intersection", None),
            ("18_16", "intersection", None),
            ("18_18", "intersection", None),
            ("18_20", "intersection", None),
            ("18_22", "intersection", None),
            ("7_19", "intersection", None),
            ("5_17", "intersection", None),
            ("4_16", "intersection", None),
            ("-2_14", "spawn", None),
            ("0_14", "intersection", None),
            ("2_14", "intersection", None),
            ("4_14", "intersection", None),
            ("6_14", "intersection", 1),
            ("8_14", "intersection", None),
            ("-2_12", "spawn", None),
            ("0_12", "intersection", None),
            ("2_12", "intersection", None),
            ("4_12", "intersection", None),
            ("6_12", "intersection", 1),
            ("8_12", "intersection", None),
            ("30_15", "spawn", None),
            ("28_15", "intersection", None),
            ("26_15", "intersection", None),
            ("24_15", "intersection", None),
            ("22_15", "intersection", None),
            ("20_15", "intersection", 2),
            ("10_13", "intersection", None),
            ("12_12", "intersection", None),
            ("14_12", "intersection", None),
            ("16_13", "intersection", 1),
            ("18_14", "intersection", None),
            ("6_16", "intersection", None),
            ("4_16", "intersection", None),
            ("2_16", "destination", None),
            ("20_12", "intersection", None),
            ("22_12", "intersection", None),
            ("24_12", "intersection", None),
            ("26_12", "destination", None),
            ("18_24", "intersection", None),
            ("18_26", "destination", None),
            ("10_11", "intersection", None),
            ("12_10", "intersection", None),
            ("14_10", "intersection", None),
            ("16_11", "intersection", 1),
            ("16_17", "intersection", None),
            ("14_18", "intersection", None),
            ("12_18", "intersection", None),
            ("10_17", "intersection", 2),
            ("16_7", "intersection", None),
            ("14_8", "intersection", None),
            ("12_8", "intersection", None),
            ("10_7", "intersection", None),
            ("10_21", "intersection", None),
            ("12_20", "intersection", None),
            ("14_20", "intersection", None),
            ("16_21", "intersection", None),
            ("18_22", "intersection", None),
            ("16_15", "intersection", None),
            ("14_16", "intersection", None),
            ("12_16", "intersection", None),
            ("10_15", "intersection", 2),
            ("6_9", "intersection", None),
            ("6_7", "intersection", None),
            ("6_5", "intersection", None),
            ("6_3", "destination", None),
            ("22_4", "spawn", None),
            ("22_6", "intersection", None),
            ("22_8", "intersection", None),
            ("22_10", "intersection", None),
            ("22_12", "intersection", None),
        ]
        for node_id, node_type, group_id in nodes_list:
            x, y = positions[node_id]
            self.add_node(node_id, x, y, node_type, group_id)

        # Define all connections based on your specification
        connections = [
            # Left path (8,x) connections
            ("8_4", "8_2"),  # 8,4 > 8,2 END
            ("8_6", "8_4"),  # 8,6 > 8,4
            ("8_8", "8_6"),  # 8,8 > 8,6
            ("8_10", "8_8"),  # 8,10 > 8,8
            ("8_12", "10_11"),  # 8,12 > 10,11 (to roundabout)
            ("8_12", "8_10"),
            ("8_14", "8_12"),  # 8,14 > 8,12
            ("8_16", "8_14"),  # 8,16 > 8,14
            ("8_18", "8_16"),  # 8,18 > 8,16
            ("8_20", "8_18"),  # 8,20 > 8,18
            ("8_22", "8_20"),  # 8,22 > 8,20
            ("8_24", "8_22"),  # 8,24 > 8,22
            ("8_26", "8_24"),  # 8,26 START > 8,24
            # Right path (18,x) connections
            ("18_2", "18_4"),  # 18,2 START > 18,4
            ("18_4", "18_6"),  # 18,4 > 18,6
            ("18_6", "18_8"),  # 18,6 > 18,8
            ("18_8", "18_10"),  # 18,8 > 18,10
            ("18_10", "18_12"),  # 18,10 > 18,12
            ("18_12", "18_14"),  # 18,12 > 18,14
            ("18_14", "18_16"),  # 18,14 > 18,16
            ("18_16", "16_17"),  # 18,16 > 16,17 (to roundabout)
            ("18_16", "18_18"),  # 18,16 > 18,18 (alternative)
            ("18_18", "18_20"),  # 18,18 > 18,20
            ("18_20", "18_22"),  # 18,20 > 18,22
            ("18_22", "18_24"),  # 18,22 > 18,24
            ("18_24", "18_26"),  # 18,24 > 18,26 END
            # Upper roundabout connections (clockwise)
            ("10_11", "12_10"),  # 10,11 > 12,10
            ("12_10", "14_10"),  # 12,10 > 14,10
            ("14_10", "16_11"),  # 14,10 > 16,11
            ("16_11", "18_12"),  # 16,11 > 18,12 UN CARRIL
            # Lower roundabout connections (counter-clockwise)
            ("16_17", "14_18"),  # 16,17 > 14,18
            ("14_18", "12_18"),  # 14,18 > 12,18
            ("12_18", "10_17"),  # 12,18 > 10,17
            ("10_17", "8_16"),  # 10,17 > 8,16 UN CARRIL
            # Uturn
            ("16_7", "14_8"),
            ("14_8", "12_8"),
            ("12_8", "10_7"),
            ("10_7", "8_6"),
            ("10_21", "12_20"),
            ("12_20", "14_20"),
            ("14_20", "16_21"),
            ("16_21", "18_22"),
            # Uturn start
            ("18_6", "16_7"),
            ("8_22", "10_21"),
            # New left exit connections
            ("6_16", "4_16"),
            ("4_16", "2_16"),
            # New right exit connections
            ("20_12", "22_12"),
            ("22_12", "24_12"),
            ("24_12", "26_12"),
            # Connections from previous nodes
            ("8_16", "6_16"),
            ("18_12", "20_12"),
            # User requested connections
            ("7_19", "5_17"),
            ("5_17", "4_16"),
            ("-2_14", "0_14"),
            ("0_14", "2_14"),
            ("2_14", "4_14"),
            ("4_14", "6_14"),
            ("6_14", "8_14"),
            ("-2_12", "0_12"),
            ("0_12", "2_12"),
            ("2_12", "4_12"),
            ("4_12", "6_12"),
            ("6_12", "8_12"),
            ("30_15", "28_15"),
            ("28_15", "26_15"),
            ("26_15", "24_15"),
            ("24_15", "22_15"),
            ("22_15", "20_15"),
            ("20_15", "18_16"),
            ("10_13", "12_12"),
            ("12_12", "14_12"),
            ("14_12", "16_13"),
            ("16_13", "18_14"),
            # Connections to existing nodes
            ("8_20", "7_19"),
            ("8_14", "10_13"),
            # User requested new path connections
            ("16_15", "14_16"),
            ("14_16", "12_16"),
            ("12_16", "10_15"),
            ("10_15", "8_14"),
            # Connection from existing node
            ("18_14", "16_15"),
            ("20_15", "18_14"),
            # New connections for user request
            ("6_9", "6_7"),
            ("6_7", "6_5"),
            ("6_5", "6_3"),
            ("8_10", "6_9"),
            # New connections for 22,x path
            ("22_4", "22_6"),
            ("22_6", "22_8"),
            ("22_8", "22_10"),
            ("22_10", "22_12"),
        ]

        # Add all connections as directed edges
        for node1, node2 in connections:
            if node1 in self.nodes and node2 in self.nodes:
                pos1 = self.nodes[node1].get_position()
                pos2 = self.nodes[node2].get_position()
                distance = np.linalg.norm(pos2 - pos1)
                # Add as directed edge (one-way)
                self.nodes[node1].add_connection(node2, distance)


class TrafficModel(ap.Model):
    """Main simulation model"""

    def setup(self):
        # Create road network
        self.road_network = RoadNetwork()
        self.road_network.create_simple_network(self.p["world_size"])

        # Initialize node occupancy
        self.node_occupancy = {node_id: None for node_id in self.road_network.nodes}

        # Initialize car list
        self.cars = ap.AgentList(self, 0, Car)

        # Statistics
        self.total_cars_spawned = 0
        self.total_cars_arrived = 0

        # Traffic light state
        self.active_group = 0  # Start with group 0
        self.group_cycle_steps = 5  # Switch every 5 steps

        # Yield rules for merge points
        self.yield_rules = [
            {"yield": "16_21", "merge": "18_22", "priority": "18_20"},
            {"yield": "5_17", "merge": "4_16", "priority": "6_16"},
            {"yield": "22_10", "merge": "22_12", "priority": "20_12"},
            {"yield": "10_7", "merge": "8_6", "priority": "8_8"},
        ]

        # Spawn-destination restrictions mapping with percentages
        # Define presets for different times of day
        self.traffic_presets = {
            "morning": {
                "-2_14": {"18_26": 294},
                "-2_12": {"6_3": 41, "8_2": 116, "26_12": 67},
                "8_26": {"2_16": 280, "18_26": 236, "6_3": 164, "8_2": 49, "26_12": 35},
                "18_2": {"8_2": 214, "18_26": 68, "2_16": 622, "6_3": 34},
                "22_4": {"26_12": 177},
                "30_15": {"18_26": 82, "2_16": 162, "6_3": 24, "8_2": 89},
            },
            "evening": {
                "-2_14": {"18_26": 444},
                "-2_12": {"6_3": 28, "8_2": 183, "26_12": 88},
                "8_26": {"2_16": 190, "18_26": 428, "6_3": 38, "8_2": 75, "26_12": 84},
                "18_2": {"8_2": 360, "18_26": 81, "2_16": 239, "6_3": 138},
                "22_4": {"26_12": 83},
                "30_15": {"18_26": 105, "2_16": 138, "6_3": 14, "8_2": 154},
            },
            "night": {
                "-2_14": {"18_26": 490},
                "-2_12": {"6_3": 41, "8_2": 226, "26_12": 134},
                "8_26": {"2_16": 243, "18_26": 609, "6_3": 52, "8_2": 94, "26_12": 133},
                "18_2": {"8_2": 219, "18_26": 76, "2_16": 470, "6_3": 179},
                "22_4": {"26_12": 131},
                "30_15": {"18_26": 108, "2_16": 179, "6_3": 7, "8_2": 165},
            },
        }

        # Set the active preset based on parameters
        preset_name = self.p.get("preset", "morning")
        if preset_name not in self.traffic_presets:
            print(
                f"Warning: Preset '{preset_name}' not found, using 'morning' as default"
            )
            preset_name = "morning"

        self.spawn_destinations = self.traffic_presets[preset_name]
        print(f"Using traffic preset: {preset_name}")

    def step(self):
        # Cycle traffic light groups every 5 steps
        if self.t % self.group_cycle_steps == 0 and self.t > 0:
            self.active_group = (self.active_group + 1) % 3
            print(f"Traffic light switched to Group {self.active_group}")

        # Spawn new cars with restricted destinations
        if (
            random.random() < self.p["spawn_rate"]
            and self.road_network.spawn_nodes
            and self.road_network.destination_nodes
        ):
            # Calculate spawn weights based on total route capacity for each spawn point
            spawn_weights = []
            valid_spawn_nodes = []

            for spawn_node in self.road_network.spawn_nodes:
                dest_probs = self.spawn_destinations.get(spawn_node, {})
                # Filter to only include destinations that exist in the network
                available_dests = {
                    dest: prob
                    for dest, prob in dest_probs.items()
                    if dest in self.road_network.destination_nodes
                }

                if available_dests:
                    # Use sum of route weights as the spawn weight for this spawn point
                    total_weight = sum(available_dests.values())
                    spawn_weights.append(total_weight)
                    valid_spawn_nodes.append(spawn_node)

            if valid_spawn_nodes:
                # Select spawn point weighted by total route capacity
                spawn_node = random.choices(
                    valid_spawn_nodes, weights=spawn_weights, k=1
                )[0]

                # Get destination probabilities for the selected spawn point
                dest_probs = self.spawn_destinations.get(spawn_node, {})
                available_dests = {
                    dest: prob
                    for dest, prob in dest_probs.items()
                    if dest in self.road_network.destination_nodes
                }

                if available_dests:
                    destinations = list(available_dests.keys())
                    probabilities = list(available_dests.values())
                    dest_node = random.choices(
                        destinations, weights=probabilities, k=1
                    )[0]
                    new_car = Car(
                        self, start_node_id=spawn_node, target_node_id=dest_node
                    )
                    self.cars.append(new_car)
                    self.total_cars_spawned += 1

                    # Debug output to show car assignments
                    if self.t % 50 == 0:  # Print every 50 steps to avoid spam
                        print(
                            f"Car {self.total_cars_spawned} spawned: {spawn_node} → {dest_node}"
                        )

        # Update all cars
        arrived_cars = []
        for car in self.cars:
            car.step()
            if car.state == "arrived":
                arrived_cars.append(car)

        # Remove arrived cars and log their completion
        for car in arrived_cars:
            print(
                f"Car completed journey: {car.path[0] if car.path else 'unknown'} → {car.target_node_id}"
            )
            # Clear occupancy (safety check - should already be cleared when car arrived)
            if self.node_occupancy[car.current_node_id] == car.id:
                self.node_occupancy[car.current_node_id] = None
            self.cars.remove(car)
            self.total_cars_arrived += 1

        # Print statistics every 100 steps
        if self.t % 100 == 0 and self.t > 0:
            print(
                f"Step {self.t}: {len(self.cars)} active cars, "
                f"{self.total_cars_spawned} spawned, "
                f"{self.total_cars_arrived} arrived"
            )


def draw_simulation(model, ax):
    """Draw the current state of the simulation"""
    ax.clear()
    ax.set_xlim(-50, model.p["world_size"])
    ax.set_ylim(0, model.p["world_size"])
    ax.set_aspect("equal")
    ax.set_title(f"Node-Based Traffic Simulation - Step {model.t}")

    # Draw network nodes
    stoplight_nodes = ["8_16", "8_14", "8_12", "18_16", "18_14", "18_12"]
    group_colors = {0: "orange", 1: "purple", 2: "cyan"}
    for node_id, node in model.road_network.nodes.items():
        if node_id in stoplight_nodes:
            # Highlight stoplight nodes with active group color
            color = group_colors.get(model.active_group, "yellow")
        elif node.group_id is not None:
            # Color based on group_id
            color = group_colors.get(
                node.group_id, "yellow"
            )  # Default to yellow if unknown group
        else:
            color = (
                "green"
                if node.node_type == "spawn"
                else "red"
                if node.node_type == "destination"
                else "blue"
            )

        ax.add_patch(Circle((node.x, node.y), 8, color=color, alpha=0.7))
        ax.text(node.x, node.y - 15, node_id, ha="center", fontsize=8)

    # Draw network edges as arrows
    yield_connections = [
        ("16_21", "18_22"),
        ("5_17", "4_16"),
        ("22_10", "22_12"),
        ("10_7", "8_6"),
    ]
    for node_id, node in model.road_network.nodes.items():
        for connected_id in node.connected_nodes:
            connected_node = model.road_network.nodes[connected_id]
            dx = connected_node.x - node.x
            dy = connected_node.y - node.y
            # Special color for yield connections
            if (node_id, connected_id) in yield_connections:
                arrow_color = "red"
            else:
                arrow_color = "k"
            ax.arrow(
                node.x,
                node.y,
                dx,
                dy,
                head_width=8,
                head_length=12,
                fc=arrow_color,
                ec=arrow_color,
                alpha=0.3,
                length_includes_head=True,
            )

    # Draw cars with spawn-based colors
    if len(model.cars) > 0:
        car_positions = np.array([car.position for car in model.cars])
        car_colors = [car.color for car in model.cars]

        ax.scatter(
            car_positions[:, 0],
            car_positions[:, 1],
            c=car_colors,
            s=40,
            zorder=5,
            edgecolors="black",
            linewidth=0.5,
        )

    # Add legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=8,
            label="Spawn Nodes",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=8,
            label="Destination Nodes",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=8,
            label="Intersection Nodes",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="orange",
            markersize=8,
            label="Stoplight (Group 0 Active)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="purple",
            markersize=8,
            label="Stoplight (Group 1 Active)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="cyan",
            markersize=8,
            label="Stoplight (Group 2 Active)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#4169E1",
            markersize=6,
            label="Cars from 8_26",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#32CD32",
            markersize=6,
            label="Cars from 18_2",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF1493",
            markersize=6,
            label="Cars from 2_14",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FFD700",
            markersize=6,
            label="Cars from 2_12",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#9932CC",
            markersize=6,
            label="Cars from 22_4",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF4500",
            markersize=6,
            label="Cars from 30_15",
        ),
        plt.Line2D(
            [0],
            [0],
            marker=">",
            color="red",
            markerfacecolor="red",
            markersize=8,
            label="Yield Connection",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")


def run_simulation_with_animation():
    """Run the simulation with animation"""
    model = TrafficModel(params)

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 10))

    def animate_step(model, ax):
        draw_simulation(model, ax)

    # Run with animation
    animation = ap.animate(model, fig, ax, animate_step)
    return model, animation


class TCPSender:
    def __init__(self, host="localhost", port=1101):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to Unity at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def send_complete_json(self, json_file_path):
        if not self.connected:
            return False
        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)

            json_str = json.dumps(data)
            size = len(json_str.encode("utf-8"))

            # Send size first (4 bytes)
            self.socket.sendall(size.to_bytes(4, byteorder="big"))
            # Then send the JSON data
            self.socket.sendall(json_str.encode("utf-8"))
            print(f"Sent complete JSON ({size} bytes)")
            return True
        except Exception as e:
            print(f"Send complete JSON failed: {e}")
            return False

    def disconnect(self):
        if self.socket:
            self.socket.close()
        self.connected = False


def send_json_file_to_unity(json_file_path, host="localhost", port=1101):
    sender = TCPSender(host, port)
    if sender.connect():
        sender.send_complete_json(json_file_path)
        sender.disconnect()


def run_simulation_and_send_json():
    frames = []
    model = TrafficModel(params)
    model.setup()
    for t in range(params["steps"]):
        model.t = t + 1  # Manually set time counter for traffic light cycling
        model.step()
        frame = {
            "time": model.t,
            "active_group": model.active_group,  # Traffic light group instead of lights
            "cars": [
                {
                    "id": car.id,  # Use persistent car ID from agentpy
                    "x": float(car.position[0]),
                    "y": float(car.position[1]),
                    "state": car.state,
                    "current_node": car.current_node_id,
                    "spawn_node": car.path[0] if car.path else "unknown",
                    "target_node": car.target_node_id,
                    "is_at_exit": car.current_node_id
                    in model.road_network.destination_nodes,
                }
                for i, car in enumerate(model.cars)
            ],
        }
        frames.append(frame)

    # Guardar
    with open("data.json", "w") as f:
        json.dump(frames, f)

    # Mandar a unity
    send_json_file_to_unity("data.json")
    print("ENVIADO A UNITY EXITOSAMENTE!!!!!! YAY")


if __name__ == "__main__":
    print("Traffic Simulation with Unity Integration")
    print("=" * 50)

    # Get preset choice
    preset_choice = input("Choose traffic preset (1-3, default: 1): ").strip()
    if preset_choice == "1" or preset_choice == "":
        params["preset"] = "morning"
    elif preset_choice == "2":
        params["preset"] = "evening"
    elif preset_choice == "3":
        params["preset"] = "night"
    else:
        print("invalid: usando mañana")
        params["preset"] = "morning"

    run_simulation_and_send_json()

    if params["animation"]:
        model, anim = run_simulation_with_animation()
        anim.save("Traffic4.gif", writer="pillow", fps=5)
