import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
from collections import defaultdict

# Simulation parameters
params = {
    "steps": 150,
    "spawn_rate": 0.15,  # Probability of spawning a car each step
    "car_speed": 100.0,  # Units per step
    "world_size": 500,  # World boundaries (0 to world_size)
}


class TrafficNode:
    """Represents a node in the road network"""

    def __init__(self, node_id, x, y, node_type="intersection"):
        self.id = node_id
        self.x = x
        self.y = y
        self.node_type = node_type  # "spawn", "intersection", "destination"
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
        self.progress = 0.0  # Progress along current edge (0.0 to 1.0)
        self.speed = params["car_speed"]
        self.state = "moving"  # 'moving', 'arrived'

        # Calculate initial path
        self.calculate_path()

        # Set initial position
        if self.path:
            start_node = self.model.road_network.nodes[self.current_node_id]
            self.position = start_node.get_position().copy()
        else:
            self.state = "arrived"

    def calculate_path(self):
        """Simple pathfinding - for now just direct connection or random walk"""
        network = self.model.road_network

        # If direct connection exists, use it
        if self.target_node_id in network.nodes[self.current_node_id].connected_nodes:
            self.path = [self.current_node_id, self.target_node_id]
        else:
            # Simple random walk pathfinding (not optimal, but works for demo)
            self.path = self.find_random_path()

    def find_random_path(self):
        """Find a path using random walk with some bias toward target"""
        network = self.model.road_network
        path = [self.current_node_id]
        current = self.current_node_id
        max_steps = 30  # Prevent infinite loops

        for _ in range(max_steps):
            if current == self.target_node_id:
                break

            connections = list(network.nodes[current].connected_nodes.keys())
            if not connections:
                break

            # Simple heuristic: prefer nodes closer to target
            target_pos = network.nodes[self.target_node_id].get_position()
            best_next = None
            best_distance = float("inf")

            for next_node_id in connections:
                next_pos = network.nodes[next_node_id].get_position()
                distance = np.linalg.norm(next_pos - target_pos)
                if distance < best_distance:
                    best_distance = distance
                    best_next = next_node_id

            if best_next and best_next not in path:  # Avoid cycles
                path.append(best_next)
                current = best_next
            else:
                # Fallback to random choice
                available = [n for n in connections if n not in path]
                if available:
                    next_node = random.choice(available)
                    path.append(next_node)
                    current = next_node
                else:
                    break

        return path

    def step(self):
        if self.state == "arrived" or not self.path:
            return

        # Check if we've reached the end of our path
        if self.current_path_index >= len(self.path) - 1:
            self.state = "arrived"
            return

        # Get current and next nodes
        current_node_id = self.path[self.current_path_index]
        next_node_id = self.path[self.current_path_index + 1]

        network = self.model.road_network
        current_node = network.nodes[current_node_id]
        next_node = network.nodes[next_node_id]

        # Calculate distance between nodes
        edge_distance = current_node.connected_nodes.get(next_node_id, 0)
        if edge_distance == 0:
            # Fallback to Euclidean distance
            edge_distance = np.linalg.norm(
                next_node.get_position() - current_node.get_position()
            )

        # Move along the edge
        if edge_distance > 0:
            progress_increment = self.speed / edge_distance
            self.progress += progress_increment

            # Update position based on progress
            start_pos = current_node.get_position()
            end_pos = next_node.get_position()
            self.position = start_pos + self.progress * (end_pos - start_pos)

            # Check if we've reached the next node
            if self.progress >= 1.0:
                self.progress = 0.0
                self.current_path_index += 1
                self.current_node_id = next_node_id
                self.position = end_pos.copy()


class RoadNetwork:
    """Manages the road network of nodes and connections"""

    def __init__(self):
        self.nodes = {}
        self.spawn_nodes = []
        self.destination_nodes = []

    def add_node(self, node_id, x, y, node_type="intersection"):
        """Add a node to the network"""
        node = TrafficNode(node_id, x, y, node_type)
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
        """Create custom road network based on specified layout"""
        # Scale factor to fit coordinates in world_size
        scale = world_size / 30  # Assuming max coordinate is around 26

        # Left path (column 8) - going from bottom to top
        self.add_node("8_2", 8 * scale, 2 * scale, "destination")  # END
        self.add_node("8_4", 8 * scale, 4 * scale, "intersection")
        self.add_node("8_6", 8 * scale, 6 * scale, "intersection")
        self.add_node("8_8", 8 * scale, 8 * scale, "intersection")
        self.add_node("8_10", 8 * scale, 10 * scale, "intersection")
        self.add_node("8_12", 8 * scale, 12 * scale, "intersection")
        self.add_node("8_14", 8 * scale, 14 * scale, "intersection")
        self.add_node("8_16", 8 * scale, 16 * scale, "intersection")
        self.add_node("8_18", 8 * scale, 18 * scale, "intersection")
        self.add_node("8_20", 8 * scale, 20 * scale, "intersection")
        self.add_node("8_22", 8 * scale, 22 * scale, "intersection")
        self.add_node("8_24", 8 * scale, 24 * scale, "intersection")
        self.add_node("8_26", 8 * scale, 26 * scale, "spawn")  # START

        # Right path (column 18) - going from top to bottom
        self.add_node("18_2", 18 * scale, 2 * scale, "spawn")  # START
        self.add_node("18_4", 18 * scale, 4 * scale, "intersection")
        self.add_node("18_6", 18 * scale, 6 * scale, "intersection")
        self.add_node("18_8", 18 * scale, 8 * scale, "intersection")
        self.add_node("18_10", 18 * scale, 10 * scale, "intersection")
        self.add_node("18_12", 18 * scale, 12 * scale, "intersection")
        self.add_node("18_14", 18 * scale, 14 * scale, "intersection")
        self.add_node("18_16", 18 * scale, 16 * scale, "intersection")
        self.add_node("18_18", 18 * scale, 18 * scale, "intersection")
        self.add_node("18_20", 18 * scale, 20 * scale, "intersection")
        self.add_node("18_22", 18 * scale, 22 * scale, "intersection")

        # New left exit path
        self.add_node("6_16", 6 * scale, 16 * scale, "intersection")
        self.add_node("4_16", 4 * scale, 16 * scale, "intersection")
        self.add_node("2_16", 2 * scale, 16 * scale, "destination")  # EXIT

        # New right exit path
        self.add_node("20_12", 20 * scale, 12 * scale, "intersection")
        self.add_node("22_12", 22 * scale, 12 * scale, "intersection")
        self.add_node("24_12", 24 * scale, 12 * scale, "intersection")
        self.add_node("26_12", 26 * scale, 12 * scale, "destination")  # EXIT
        self.add_node("18_24", 18 * scale, 24 * scale, "intersection")
        self.add_node("18_26", 18 * scale, 26 * scale, "destination")  # END

        # Roundabout/connecting nodes
        self.add_node("10_11", 10 * scale, 11 * scale, "intersection")
        self.add_node("12_10", 12 * scale, 10 * scale, "intersection")
        self.add_node("14_10", 14 * scale, 10 * scale, "intersection")
        self.add_node("16_11", 16 * scale, 11 * scale, "intersection")
        self.add_node("16_17", 16 * scale, 17 * scale, "intersection")
        self.add_node("14_18", 14 * scale, 18 * scale, "intersection")
        self.add_node("12_18", 12 * scale, 18 * scale, "intersection")
        self.add_node("10_17", 10 * scale, 17 * scale, "intersection")

        # Uturns
        self.add_node("16_7", 16 * scale, 7 * scale, "intersection")
        self.add_node("14_8", 14 * scale, 8 * scale, "intersection")
        self.add_node("12_8", 12 * scale, 8 * scale, "intersection")
        self.add_node("10_7", 10 * scale, 7 * scale, "intersection")
        self.add_node("10_21", 10 * scale, 21 * scale, "intersection")
        self.add_node("12_20", 12 * scale, 20 * scale, "intersection")
        self.add_node("14_20", 14 * scale, 20 * scale, "intersection")
        self.add_node("16_21", 16 * scale, 21 * scale, "intersection")
        self.add_node("18_22", 18 * scale, 22 * scale, "intersection")

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
            ("16_11", "18_12"),  # 16,11 > 18,12
            # Lower roundabout connections (counter-clockwise)
            ("16_17", "14_18"),  # 16,17 > 14,18
            ("14_18", "12_18"),  # 14,18 > 12,18
            ("12_18", "10_17"),  # 12,18 > 10,17
            ("10_17", "8_16"),  # 10,17 > 8,16
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
            ("10_17", "6_16"),
            ("16_11", "20_12"),
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
        self.road_network.create_simple_network(params["world_size"])

        # Initialize car list
        self.cars = ap.AgentList(self, 0, Car)

        # Statistics
        self.total_cars_spawned = 0
        self.total_cars_arrived = 0

    def step(self):
        # Spawn new cars with random destinations
        if (
            random.random() < params["spawn_rate"]
            and self.road_network.spawn_nodes
            and self.road_network.destination_nodes
        ):
            spawn_node = random.choice(self.road_network.spawn_nodes)

            # Assign a random exit point (destination node) to each car
            available_destinations = [
                dest
                for dest in self.road_network.destination_nodes
                if dest != spawn_node
            ]  # Don't spawn to same location

            if available_destinations:
                dest_node = random.choice(available_destinations)
                new_car = Car(self, start_node_id=spawn_node, target_node_id=dest_node)
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
    ax.set_xlim(0, params["world_size"])
    ax.set_ylim(0, params["world_size"])
    ax.set_aspect("equal")
    ax.set_title(f"Node-Based Traffic Simulation - Step {model.t}")

    # Draw network nodes
    for node_id, node in model.road_network.nodes.items():
        color = (
            "green"
            if node.node_type == "spawn"
            else "red"
            if node.node_type == "destination"
            else "blue"
        )

        ax.add_patch(Circle((node.x, node.y), 8, color=color, alpha=0.7))
        ax.text(node.x, node.y - 15, node_id, ha="center", fontsize=8)

    # Draw network edges
    for node_id, node in model.road_network.nodes.items():
        for connected_id in node.connected_nodes:
            connected_node = model.road_network.nodes[connected_id]
            ax.plot(
                [node.x, connected_node.x],
                [node.y, connected_node.y],
                "k-",
                alpha=0.3,
                linewidth=1,
            )

    # Draw cars with color coding based on destination
    if len(model.cars) > 0:
        car_positions = np.array([car.position for car in model.cars])

        # Color cars based on their target destination
        car_colors = []
        destination_color_map = {
            "8_2": "orange",
            "18_26": "purple",
            "2_16": "yellow",
            "26_12": "cyan",
        }

        for car in model.cars:
            color = destination_color_map.get(car.target_node_id, "orange")
            car_colors.append(color)

        ax.scatter(
            car_positions[:, 0],
            car_positions[:, 1],
            c=car_colors,
            s=40,
            zorder=5,
            edgecolors="black",
            linewidth=0.5,
        )

    # Add enhanced legend with destination info
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
            markersize=6,
            label="Cars → 8_2",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="purple",
            markersize=6,
            label="Cars → 18_26",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="yellow",
            markersize=6,
            label="Cars → 2_16",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="cyan",
            markersize=6,
            label="Cars → 26_12",
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


def run_simulation_headless(steps=None):
    """Run simulation without animation for data extraction"""
    if steps is None:
        steps = params["steps"]

    model = TrafficModel(params)
    model.run(steps=steps)
    return model


def get_simulation_data(model):
    """Extract current simulation state for Unity"""
    data = {
        "step": model.t,
        "cars": [],
        "nodes": {},
        "stats": {
            "active_cars": len(model.cars),
            "total_spawned": model.total_cars_spawned,
            "total_arrived": model.total_cars_arrived,
        },
    }

    # Car positions and states
    for i, car in enumerate(model.cars):
        data["cars"].append(
            {
                "id": i,
                "x": float(car.position[0]),
                "y": float(car.position[1]),
                "state": car.state,
                "target_node": car.target_node_id,
            }
        )

    # Node positions and connections
    for node_id, node in model.road_network.nodes.items():
        data["nodes"][node_id] = {
            "x": node.x,
            "y": node.y,
            "type": node.node_type,
            "connections": list(node.connected_nodes.keys()),
        }

    return data


# Example usage:
if __name__ == "__main__":
    print("Starting Node-Based Traffic Simulation")
    print("=" * 50)
    print("Network Configuration:")
    print("- Custom road layout with directed paths")
    print("- START: (8,26) and (18,2)")
    print("- EXIT POINTS: (8,2), (18,26), (2,16), (26,12)")
    print("- Cars assigned random exit destinations")
    print("=" * 50)

    # Choose your preferred way to run:

    # Option 1: Headless for clean output (no animation warning)
    # print("Running headless simulation for 200 steps...")
    # model = run_simulation_headless(steps=200)
    # data = get_simulation_data(model)
    # print(f"Final state: {data['stats']}")

    # Option 2: With animation (will show the harmless warning)
    print("Running with animation...")
    model, anim = run_simulation_with_animation()
    plt.show()

    # Run the simulation

    # To save as GIF (uncomment the line below):
    anim.save("Traffic.gif", writer="pillow", fps=5)

    # Print some sample data for Unity integration
    # print("\nSample data structure for Unity:")
    # print(f"Number of nodes: {len(data['nodes'])}")
    # print(f"Sample car data: {data['cars'][:2] if data['cars'] else 'No cars active'}")
    # print(f"Sample node data: {list(data['nodes'].items())[:2]}")
