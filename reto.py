import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
from collections import defaultdict

# Simulation parameters
params = {
    "steps": 50,
    "spawn_rate": 0.15,  # Probability of spawning a car each step
    "car_speed": 0.0,  # Units per step
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
        max_steps = 10  # Prevent infinite loops

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
        """Create a simple road network for demonstration"""
        # Create a simple grid-like network
        self.add_node("spawn_n", world_size / 2, world_size - 50, "spawn")
        self.add_node("spawn_s", world_size / 2, 50, "spawn")
        self.add_node("spawn_e", world_size - 50, world_size / 2, "spawn")
        self.add_node("spawn_w", 50, world_size / 2, "spawn")

        # Central intersection
        self.add_node("center", world_size / 2, world_size / 2, "intersection")

        # Intermediate nodes
        self.add_node("north_mid", world_size / 2, world_size * 0.75, "intersection")
        self.add_node("south_mid", world_size / 2, world_size * 0.25, "intersection")
        self.add_node("east_mid", world_size * 0.75, world_size / 2, "intersection")
        self.add_node("west_mid", world_size * 0.25, world_size / 2, "intersection")

        # Destination nodes
        self.add_node("dest_ne", world_size * 0.8, world_size * 0.8, "destination")
        self.add_node("dest_nw", world_size * 0.2, world_size * 0.8, "destination")
        self.add_node("dest_se", world_size * 0.8, world_size * 0.2, "destination")
        self.add_node("dest_sw", world_size * 0.2, world_size * 0.2, "destination")

        # Connect the network
        connections = [
            ("spawn_n", "north_mid"),
            ("north_mid", "center"),
            ("center", "south_mid"),
            ("south_mid", "spawn_s"),
            ("spawn_e", "east_mid"),
            ("east_mid", "center"),
            ("center", "west_mid"),
            ("west_mid", "spawn_w"),
            ("north_mid", "dest_ne"),
            ("north_mid", "dest_nw"),
            ("south_mid", "dest_se"),
            ("south_mid", "dest_sw"),
            ("east_mid", "dest_ne"),
            ("east_mid", "dest_se"),
            ("west_mid", "dest_nw"),
            ("west_mid", "dest_sw"),
        ]

        for node1, node2 in connections:
            self.add_edge(node1, node2)


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
        # Spawn new cars
        if (
            random.random() < params["spawn_rate"]
            and self.road_network.spawn_nodes
            and self.road_network.destination_nodes
        ):
            spawn_node = random.choice(self.road_network.spawn_nodes)
            dest_node = random.choice(self.road_network.destination_nodes)

            if spawn_node != dest_node:  # Don't spawn car to same location
                new_car = Car(self, start_node_id=spawn_node, target_node_id=dest_node)
                self.cars.append(new_car)
                self.total_cars_spawned += 1

        # Update all cars
        arrived_cars = []
        for car in self.cars:
            car.step()
            if car.state == "arrived":
                arrived_cars.append(car)

        # Remove arrived cars
        for car in arrived_cars:
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

    # Draw cars
    if len(model.cars) > 0:
        car_positions = np.array([car.position for car in model.cars])
        ax.scatter(
            car_positions[:, 0],
            car_positions[:, 1],
            c="orange",
            s=30,
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
            markersize=6,
            label="Cars",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")


def run_simulation():
    """Run the simulation with animation"""
    model = TrafficModel(params)

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 10))

    def animate_step(model, ax):
        draw_simulation(model, ax)

    # Run with animation
    animation = ap.animate(model, fig, ax, animate_step)
    return model, animation


# Example usage:
if __name__ == "__main__":
    print("Starting Node-Based Traffic Simulation")
    print("=" * 50)
    print("Network Configuration:")
    print("- 4 spawn nodes (edges)")
    print("- 4 destination nodes (corners)")
    print("- 5 intersection nodes")
    print("- Bidirectional connections")
    print("=" * 50)

    # Run the simulation
    model, anim = run_simulation()

    # To save as GIF (uncomment the line below):
    anim.save("node_traffic_simulation.gif", writer="pillow", fps=10)

    # To display in Jupyter (uncomment the lines below):
    # from IPython.display import HTML
    # HTML(anim.to_jshtml())

    plt.show()
