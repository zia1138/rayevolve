"""Evaluator for the Traveling Salesman Problem."""

import sys
import uuid
import json
import importlib.util
from pathlib import Path

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


def load_data(filepath):
    """
    Load TSP instance data from a file.

    Args:
        filepath: Path to the TSP file

    Returns:
        List containing a single dictionary with key 'nodes'.
    """
    import re

    nodes = []
    dimension = 0
    reading_nodes = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())

            elif line == "NODE_COORD_SECTION":
                reading_nodes = True

            elif line == "EOF":
                reading_nodes = False

            elif reading_nodes and line:
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 3:
                    try:
                        x, y = float(parts[1]), float(parts[2])
                        nodes.append([x, y])
                    except ValueError:
                        continue

    if dimension > 0 and len(nodes) != dimension:
        print(f"Warning: Expected {dimension} nodes but found {len(nodes)}")

    return [{'nodes': nodes}]


def eval_func(nodes, tour):
    """
    Evaluate a predicted TSP tour.

    Args:
        nodes : (list) List of (x, y) coordinates representing cities.
                Format: [(x1, y1), (x2, y2), ..., (xn, yn)]
        tour  : (list) Predicted tour from the solver as list of node indices.
                Format: [0, 3, 1, ...]

    Returns:
        float: The total tour cost (sum of Euclidean distances).

    Raises:
        Exception if the tour is invalid.
    """
    import math

    num_nodes = len(nodes)

    if len(tour) != num_nodes:
        raise Exception(f"Invalid tour length: Expected {num_nodes}, got {len(tour)}")
    nodes_set = set(tour)

    if len(nodes_set) != num_nodes:
        raise Exception(f"Invalid tour: Contains {len(nodes_set)} unique nodes, expected {num_nodes}")

    expected_nodes = set(range(num_nodes))
    if nodes_set != expected_nodes:
        raise Exception(f"Invalid tour: Contains out-of-range or missing nodes")

    def calculate_tour_cost(nodes, tour):
        cost = 0
        for i in range(len(tour)):
            from_node = tour[i]
            to_node = tour[(i + 1) % len(tour)]

            from_x, from_y = nodes[from_node]
            to_x, to_y = nodes[to_node]
            segment_cost = math.sqrt((to_x - from_x) ** 2 + (to_y - from_y) ** 2)

            cost += segment_cost

        return cost

    pred_cost = calculate_tour_cost(nodes, tour)

    return pred_cost


OPTIMAL_SCORES = {
    'valid_instances/E1k.0': [23360648.0],
    'valid_instances/E1k.1': [22985695.0],
    'valid_instances/E1k.2': [23023351.0],
    'valid_instances/E1k.3': [23149856.0],
    'valid_instances/E1k.4': [22698717.0],
    'valid_instances/E1k.5': [23192391.0],
    'valid_instances/E1k.6': [23349803.0],
    'valid_instances/E1k.7': [22882343.0],
    'valid_instances/E1k.8': [23027023.0],
    'valid_instances/E1k.9': [23356256.0],
    'hard_test_instances/C100k.0': [104617752.0],
    'hard_test_instances/C100k.1': [105390777.0],
    'hard_test_instances/C10k.0': [33001034.0],
    'hard_test_instances/C10k.1': [33186248.0],
    'hard_test_instances/C10k.2': [33155424.0],
    'hard_test_instances/C316k.0': [186870839.0],
    'hard_test_instances/C31k.0': [59545390.0],
    'hard_test_instances/C31k.1': [59293266.0],
    'hard_test_instances/E100k.0': [225784127.0],
    'hard_test_instances/E100k.1': [225654639.0],
    'hard_test_instances/E10M.0': [2253040346.0],
    'hard_test_instances/E10k.0': [71865826.0],
    'hard_test_instances/E10k.1': [72031630.0],
    'hard_test_instances/E10k.2': [71822483.0],
    'hard_test_instances/E1M.0': [713187688.0],
    'hard_test_instances/E316k.0': [401301206.0],
    'hard_test_instances/E31k.0': [127282138.0],
    'hard_test_instances/E31k.1': [127452384.0],
    'hard_test_instances/E3M.0': [1267295473.0],
    'easy_test_instances/brd14051.tsp': [469385.0],
    'easy_test_instances/d1291.tsp': [50801.0],
    'easy_test_instances/d15112.tsp': [1573084.0],
    'easy_test_instances/d1655.tsp': [62128.0],
    'easy_test_instances/d18512.tsp': [645238.0],
    'easy_test_instances/d2103.tsp': [80450.0],
    'easy_test_instances/fl1400.tsp': [20127.0],
    'easy_test_instances/fl1577.tsp': [22249.0],
    'easy_test_instances/fl3795.tsp': [28772.0],
    'easy_test_instances/fnl4461.tsp': [182566.0],
    'easy_test_instances/nrw1379.tsp': [56638.0],
    'easy_test_instances/pcb1173.tsp': [56892.0],
    'easy_test_instances/pcb3038.tsp': [137694.0],
    'easy_test_instances/pr1002.tsp': [259045.0],
    'easy_test_instances/pr2392.tsp': [378032.0],
    'easy_test_instances/rl11849.tsp': [923288.0],
    'easy_test_instances/rl1304.tsp': [252948.0],
    'easy_test_instances/rl1323.tsp': [270199.0],
    'easy_test_instances/rl1889.tsp': [316536.0],
    'easy_test_instances/rl5915.tsp': [565530.0],
    'easy_test_instances/rl5934.tsp': [556045.0],
    'easy_test_instances/u1060.tsp': [224094.0],
    'easy_test_instances/u1432.tsp': [152970.0],
    'easy_test_instances/u1817.tsp': [57201.0],
    'easy_test_instances/u2152.tsp': [64253.0],
    'easy_test_instances/u2319.tsp': [234256.0],
    'easy_test_instances/usa13509.tsp': [19982859.0],
    'easy_test_instances/vm1084.tsp': [239297.0],
    'easy_test_instances/vm1748.tsp': [336556.0],
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
        )
        score = eval_func(
            nodes=instance["nodes"],
            tour=solution["tour"],
        )
        return {"score": score, "error": None}
    except Exception as e:
        return {"score": str(e), "error": str(e)}


def evaluate_candidate(main_py_path: str | Path, data_dir: Path, test: bool = False) -> dict:
    """Evaluate a candidate main.py against TSP instances.

    By default, evaluates on dev instances (valid_instances/).
    If test=True, evaluates on easy_test_instances/ + hard_test_instances/.

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
            instance_id = f"{instance_file.stem}_{idx}"
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
