import numpy as np
import networkx as nx


def generate_graph(n, type="Watts-Strogatz", seed=42):
    """Generate a connected graph of specified type and size n."""
    if type == "Complete":
        G = nx.complete_graph(n)
    elif type == "Watts-Strogatz":
        k = 4  # Average degree for Watts-Strogatz
        p = 0.4  # Rewiring probability
        G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    elif type == "2D Grid":
        length, width = best_side_from_surface(n)
        if length == -1:
            raise ValueError("n must be a composite number for 2D Grid.")
        else:
            G = nx.grid_2d_graph(length, width)
        G = nx.convert_node_labels_to_integers(G)
    elif type == "Cycle":
        G = nx.cycle_graph(n)
    elif type == "Geometric":
        G = generate_connected_rgg(n)
    else:
        raise ValueError("Wrong graph type.")

    # check if graph is connected
    if not nx.is_connected(G):
        print("Graph is not connected. Generating a new graph.")
        return generate_graph(n, type, seed + 1)
    else:
        return G


def best_side_from_surface(S):
    """Find the best dimensions for the 2D Grid graph."""
    root = int(S**0.5)
    for i in range(root, 0, -1):
        if S % i == 0:
            j = S // i
            return (i, j)
    return -1, -1


def generate_connected_rgg(n, c=8, max_attempts=100):
    """Generate a connected random geometric graph with n nodes."""
    radius = np.sqrt((np.log(n) + c) / (np.pi * n))
    print("radius", np.round(radius, 2))
    for attempt in range(max_attempts):
        G = nx.random_geometric_graph(n=n, radius=radius, seed=42)
        if nx.is_connected(G):
            print(f"Connected graph found after {attempt + 1} attempt(s)")
            return G
    raise ValueError("Failed to generate a connected graph. Try increasing the radius.")
