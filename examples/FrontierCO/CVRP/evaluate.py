"""Evaluator for the Capacitated Vehicle Routing Problem (CVRP)."""

import sys
import uuid
import json
import math
import importlib.util
from pathlib import Path

import numpy as np
import typer


def load_module_from_path(file_path: str | Path, unique: bool = True):
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    module_name = path.stem
    if unique:
        module_name = f"{module_name}_{uuid.uuid4().hex}"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def read_vrp_file(filename):
    """Read CVRP Problem instance from a .vrp file."""
    with open(filename, 'r') as f:
        lines = f.readlines()

        # Parse metadata from header
        n = None  # Number of nodes (including depot)
        capacity = None
        coordinates = []
        demands = []
        depot_idx = 0  # Default depot index - node 1 (0-indexed) is standard default
        depot_found = False

        # Read through file sections
        section = None
        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for section headers
            if line.startswith("DIMENSION"):
                n = int(line.split(":")[1].strip())
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(":")[1].strip())
            elif line == "NODE_COORD_SECTION":
                section = "coords"
                continue
            elif line == "DEMAND_SECTION":
                section = "demand"
                continue
            elif line == "DEPOT_SECTION":
                section = "depot"
                depot_found = True
                continue
            elif line == "EOF":
                break

            # Parse data based on current section
            if section == "coords":
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0]) - 1  # Convert to 0-indexed
                    x = float(parts[1])
                    y = float(parts[2])

                    # Ensure we have a spot in the array for this node
                    while len(coordinates) <= node_id:
                        coordinates.append(None)

                    coordinates[node_id] = (x, y)

            elif section == "demand":
                parts = line.split()
                if len(parts) >= 2:
                    node_id = int(parts[0]) - 1  # Convert to 0-indexed
                    demand = int(parts[1])

                    # Ensure we have a spot in the array for this demand
                    while len(demands) <= node_id:
                        demands.append(None)

                    demands[node_id] = demand

            elif section == "depot":
                try:
                    depot = int(line)
                    if depot > 0:  # Valid depot ID
                        depot_idx = depot - 1  # Convert to 0-indexed
                except ValueError:
                    pass  # Skip if not a valid depot ID

        # Ensure all node coordinates and demands are loaded
        if n is None:
            n = len(coordinates)

        # Handle special cases for depot identification
        if not depot_found:
            # Check if there's a node with zero demand - that's often the depot
            for i, demand in enumerate(demands):
                if demand == 0:
                    depot_idx = i
                    break
            # Otherwise, use node 0 as default depot (common convention)

        # Calculate distance matrix
        coords = np.array(coordinates)

        # Calculate pairwise squared differences
        x_diff = coords[:, 0, np.newaxis] - coords[:, 0]
        y_diff = coords[:, 1, np.newaxis] - coords[:, 1]

        # Calculate Euclidean distances
        dist_matrix = np.sqrt(x_diff ** 2 + y_diff ** 2)

        # Set diagonal to zero (optional, as it should already be zero)
        np.fill_diagonal(dist_matrix, 0)

    return n, capacity, coordinates, demands, dist_matrix, depot_idx


def load_data(file_path):
    """
    Load CVRP instances from .vrp files.

    Args:
        file_path (str): Path to the file containing CVRP instances

    Returns:
        list: List of dictionaries, each containing a CVRP instance with:
            - 'nodes': List of (x, y) coordinates
            - 'demands': List of customer demands
            - 'capacity': Vehicle capacity
            - 'depot_idx': Index of the depot (typically 0)
            - 'dist_matrix': Distance matrix (numpy array)
            - 'optimal_routes': List of optimal routes (if available)
    """
    instances = []

    try:
        n, capacity, coordinates, demands, dist_matrix, depot_idx = read_vrp_file(file_path)

        # Create a dictionary for this instance
        instance = {
            'nodes': coordinates,
            'demands': demands,
            'capacity': capacity,
            'depot_idx': depot_idx,
            'dist_matrix': dist_matrix,
            'optimal_routes': None  # Typically not available in .vrp files
        }

        instances.append(instance)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return instances


def validate_cvrp_solution(nodes, demands, capacity, depot_idx, routes):
    """
    Validate that a CVRP solution meets all constraints.

    Args:
        nodes (list): List of (x, y) coordinates
        demands (list): List of customer demands
        capacity (int): Vehicle capacity
        depot_idx (int): Index of the depot
        routes (list): List of routes to validate

    Raises:
        Exception: If the solution is invalid
    """
    num_nodes = len(nodes)
    all_visited = set()

    for route_idx, route in enumerate(routes):
        # Check that route starts and ends at depot
        if route[0] != depot_idx or route[-1] != depot_idx:
            raise Exception(f"Route {route_idx} does not start and end at the depot")

        # Check capacity constraint
        route_demand = sum(demands[i] for i in route[1:-1])  # Exclude depot
        if route_demand > capacity:
            raise Exception(f"Route {route_idx} exceeds capacity: {route_demand} > {capacity}")

        # Check that nodes are valid indices
        for node in route:
            if node < 0 or node >= num_nodes:
                raise Exception(f"Invalid node index {node} in route {route_idx}")

            # Add to visited set (excluding depot)
            if node != depot_idx:
                all_visited.add(node)

    # Check that all customers are visited exactly once
    expected_visited = set(range(num_nodes))
    expected_visited.remove(depot_idx)  # Exclude depot

    if all_visited != expected_visited:
        missing = expected_visited - all_visited
        duplicate = all_visited - expected_visited

        if missing:
            raise Exception(f"Nodes not visited: {missing}")
        if duplicate:
            raise Exception(f"Nodes visited more than once: {duplicate}")


def calculate_total_distance(nodes, routes):
    """
    Calculate the total distance of a CVRP solution.

    Args:
        nodes (list): List of (x, y) coordinates
        routes (list): List of routes

    Returns:
        float: Total distance of all routes
    """
    total_distance = 0

    for route in routes:
        route_distance = 0
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]

            # Calculate Euclidean distance
            from_x, from_y = nodes[from_node]
            to_x, to_y = nodes[to_node]
            segment_distance = math.sqrt((to_x - from_x) ** 2 + (to_y - from_y) ** 2)

            route_distance += segment_distance

        total_distance += route_distance

    return total_distance


def eval_func(nodes, demands, capacity, depot_idx, optimal_routes, routes):
    """
    Evaluate a predicted CVRP solution against optimal routes or calculate total distance.

    Args:
        nodes (list): List of (x, y) coordinates representing locations.
        demands (list): List of customer demands.
        capacity (int): Vehicle capacity.
        depot_idx (int): Index of the depot (typically 0).
        optimal_routes (list): Reference optimal routes (may be None if not available).
        routes (list): Predicted routes from the solver.

    Returns:
        float: Optimality gap percentage if optimal_routes is provided,
              or just the predicted solution's total distance.
    """
    # Validate solution
    validate_cvrp_solution(nodes, demands, capacity, depot_idx, routes)

    # Calculate the predicted solution cost (total distance)
    pred_cost = calculate_total_distance(nodes, routes)

    # If optimal routes are provided, calculate optimality gap
    if optimal_routes:
        opt_cost = calculate_total_distance(nodes, optimal_routes)
        opt_gap = ((pred_cost / opt_cost) - 1) * 100
        return opt_gap

    # Otherwise, just return the predicted cost
    return pred_cost


OPTIMAL_SCORES = {
    'easy_test_instances/Golden_13.vrp': [857.188745],
    'easy_test_instances/Golden_17.vrp': [707.755935],
    'easy_test_instances/Golden_10.vrp': [735.43],
    'easy_test_instances/Golden_19.vrp': [1365.6],
    'easy_test_instances/Golden_7.vrp': [10023.844627],
    'easy_test_instances/Golden_3.vrp': [10785.779388],
    'easy_test_instances/Golden_1.vrp': [5370.545835],
    'easy_test_instances/Golden_8.vrp': [11486.585777],
    'easy_test_instances/Golden_12.vrp': [1100.67],
    'easy_test_instances/Golden_5.vrp': [6460.979519],
    'easy_test_instances/Golden_18.vrp': [995.13],
    'easy_test_instances/Golden_9.vrp': [579.7],
    'easy_test_instances/Golden_11.vrp': [911.98],
    'easy_test_instances/Golden_4.vrp': [13541.657456],
    'easy_test_instances/Golden_16.vrp': [1611.28],
    'easy_test_instances/Golden_6.vrp': [8348.949187],
    'easy_test_instances/Golden_15.vrp': [1337.27],
    'easy_test_instances/Golden_2.vrp': [8205.866802],
    'easy_test_instances/Golden_20.vrp': [1817.59],
    'easy_test_instances/Golden_14.vrp': [1080.55],
    'hard_test_instances/Leuven1.vrp': [192848.0],
    'hard_test_instances/Leuven2.vrp': [111391.0],
    'hard_test_instances/Antwerp1.vrp': [477277.0],
    'hard_test_instances/Antwerp2.vrp': [291350.0],
    'hard_test_instances/Ghent1.vrp': [469531.0],
    'hard_test_instances/Ghent2.vrp': [257748.0],
    'hard_test_instances/Brussels1.vrp': [501719.0],
    'hard_test_instances/Brussels2.vrp': [345468.0],
    'hard_test_instances/Flanders1.vrp': [7240118.0],
    'hard_test_instances/Flanders2.vrp': [4373244.0],
    'valid_instances/instance_100_5.vrp': [1269.090891],
    'valid_instances/instance_100_1.vrp': [1322.220351],
    'valid_instances/instance_100_2.vrp': [1273.835052],
    'valid_instances/instance_100_3.vrp': [1239.690157],
    'valid_instances/instance_100_4.vrp': [1289.129098],
    'valid_instances/instance_50_1.vrp': [753.946825],
    'valid_instances/instance_50_3.vrp': [826.843367],
    'valid_instances/instance_50_5.vrp': [818.44171],
    'valid_instances/instance_50_4.vrp': [776.674785],
    'valid_instances/instance_20_1.vrp': [453.252768],
    'valid_instances/instance_50_2.vrp': [743.427813],
    'valid_instances/instance_20_3.vrp': [487.1528],
    'valid_instances/instance_20_4.vrp': [463.300497],
    'valid_instances/instance_20_5.vrp': [388.950037],
    'valid_instances/instance_20_2.vrp': [455.784836],
}


def norm_score_for_instance(instance_key, score):
    optimal_list = OPTIMAL_SCORES.get(instance_key)
    if optimal_list is None or not isinstance(score, (int, float)):
        return None
    optimal = optimal_list[0]
    return 1 - abs(score - optimal) / max(score, optimal)


def _solve_and_eval(main_py_path, instance_id, instance):
    """Load candidate module and run solve + eval for a single instance."""
    try:
        candidate = load_module_from_path(main_py_path)
        solution = candidate.solve(
            instance_id=instance_id,
            nodes=instance["nodes"],
            demands=instance["demands"],
            capacity=instance["capacity"],
            depot_idx=instance["depot_idx"],
        )
        score = eval_func(
            nodes=instance["nodes"],
            demands=instance["demands"],
            capacity=instance["capacity"],
            depot_idx=instance["depot_idx"],
            optimal_routes=instance["optimal_routes"],
            routes=solution["routes"],
        )
        return {"score": score, "error": None}
    except Exception as e:
        return {"score": str(e), "error": str(e)}


def evaluate_candidate(main_py_path: str | Path, data_dir: Path, test: bool = False) -> dict:
    """Evaluate a candidate main.py against CVRP instances.

    By default, evaluates on dev instances (valid_instances/).
    If test=True, evaluates on test instances (easy_test_instances/ + hard_test_instances/).

    Each instance is evaluated sequentially.
    """
    main_py_path = str(Path(main_py_path).resolve())

    if test:
        prefixes = ("easy_test_instances/", "hard_test_instances/")
    else:
        prefixes = ("valid_instances/",)

    instance_keys = sorted(k for k in OPTIMAL_SCORES if any(k.startswith(p) for p in prefixes))

    all_normed_scores = []

    for instance_key in instance_keys:
        instance_file = data_dir / instance_key
        all_instances = load_data(str(instance_file))

        for idx, instance in enumerate(all_instances):
            instance_id = f"{instance_file.name}_{idx}"
            result = _solve_and_eval(main_py_path, instance_id, instance)

            if result["error"] is not None:
                all_normed_scores.append(0.0)
                print(f"[instance_id={instance_id}], error=true, normalized_score=0.0000, {result['error']}")
            else:
                normed = norm_score_for_instance(instance_key, result["score"])
                normed = normed if normed is not None else 0.0
                all_normed_scores.append(normed)
                print(f"[instance_id={instance_id}], error=false, normalized_score={normed:.4f}")

    combined_score = sum(all_normed_scores) / len(all_normed_scores) if all_normed_scores else 0.0

    return {"correct": True, "error": "", "combined_score": combined_score}


app = typer.Typer()


@app.command()
def main(
    test: bool = typer.Option(False, "--test", help="Evaluate on test instances instead of dev instances"),
):
    data_dir = Path(__file__).parent
    result = evaluate_candidate(data_dir / "main.py", data_dir, test=test)

    print(f"Average normalized score: {result['combined_score']:.4f}")

    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    app()
