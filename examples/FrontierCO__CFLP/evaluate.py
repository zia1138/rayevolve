"""Evaluator for the FrontierCO Capacitated Facility Location Problem."""

import sys
import uuid
import json
import logging
import importlib.util
from pathlib import Path

import ray
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


def load_data(input_filename):
    """
    Reads a Capacitated Facility Location Problem instance from a file.

    File format (.plc / .txt):
        Line 1: n m          (number of facilities, number of customers)
        Next n lines: b_i f_i  (capacity and fixed cost for facility i)
        Next values: d_1 d_2 ... d_m  (customer demands, may span multiple lines)
        Next n*m values: c_ij  (transport cost from facility i to customer j,
                                row-major: facility 1 to all customers, then facility 2, etc.)

    Returns:
        A list containing a single dict with keys:
            n, m, capacities, fixed_cost, demands, trans_costs
    """
    with open(input_filename, "r") as f:
        tokens = f.read().split()

    idx = 0
    n = int(tokens[idx]); idx += 1
    m = int(tokens[idx]); idx += 1

    capacities = []
    fixed_cost = []
    for _ in range(n):
        capacities.append(float(tokens[idx])); idx += 1
        fixed_cost.append(float(tokens[idx])); idx += 1

    demands = []
    for _ in range(m):
        demands.append(float(tokens[idx])); idx += 1

    trans_costs = []
    for _ in range(n):
        row = []
        for _ in range(m):
            row.append(float(tokens[idx])); idx += 1
        trans_costs.append(row)

    return [{
        "n": n,
        "m": m,
        "capacities": capacities,
        "fixed_cost": fixed_cost,
        "demands": demands,
        "trans_costs": trans_costs,
    }]


def eval_func(n, m, capacities, fixed_cost, demands, trans_costs,
              total_cost, facilities_open, assignments, **kwargs):
    """
    Evaluates a proposed CFLP solution.

    Matches the FrontierCO evaluation exactly: weighted-average transport cost per customer,
    with 1e-6 tolerance on demand satisfaction and capacity constraints.

    Returns:
        The computed total cost (float) if the solution is feasible.

    Raises:
        ValueError with an informative message if any constraint is violated.
    """
    computed_total_cost = 0.0

    # Fixed costs for open facilities
    for i in range(n):
        if facilities_open[i] == 1:
            computed_total_cost += fixed_cost[i]

    # Assignment cost for each customer (weighted average)
    for j in range(m):
        customer_demand = demands[j]
        allocated_amount = sum(assignments[j])

        if abs(allocated_amount - customer_demand) > 1e-6:
            raise ValueError(
                f"Customer {j} demand violation: total assigned {allocated_amount} "
                f"does not equal demand {customer_demand}."
            )

        weighted_cost = 0.0
        for i in range(n):
            allocation = assignments[j][i]

            if allocation < 0:
                raise ValueError(
                    f"Customer {j} has negative allocation {allocation} for facility {i}."
                )

            if allocation > 0 and facilities_open[i] != 1:
                raise ValueError(
                    f"Customer {j} has allocation {allocation} for facility {i}, which is closed."
                )

            fraction = allocation / customer_demand if customer_demand > 0 else 0.0
            weighted_cost += fraction * trans_costs[i][j]

        computed_total_cost += weighted_cost

    # Capacity constraints
    assigned_demand = [0.0] * n
    for i in range(n):
        for j in range(m):
            assigned_demand[i] += assignments[j][i]

    for i in range(n):
        if assigned_demand[i] > capacities[i] + 1e-6:
            excess = assigned_demand[i] - capacities[i]
            raise ValueError(
                f"Facility {i} exceeds capacity by {excess} units."
            )

    return computed_total_cost


# Optimal scores from FrontierCO benchmark (best-known solutions).
# Each file contains one instance, so each value is a single-element list.
OPTIMAL_SCORES = {
    # Validation instances (100 facilities x 100 customers)
    "cflp_corn_n100_m100_r5.0_1.txt": [17769.059032386],
    "cflp_corn_n100_m100_r5.0_2.txt": [17458.209008233],
    "cflp_corn_n100_m100_r5.0_3.txt": [18172.936618284],
    "cflp_corn_n100_m100_r5.0_4.txt": [18555.250515996],
    "cflp_corn_n100_m100_r5.0_5.txt": [17202.336476137],
    "cflp_corn_n100_m100_r5.0_6.txt": [17316.344512835],
    "cflp_corn_n100_m100_r5.0_7.txt": [17305.736933955],
    "cflp_corn_n100_m100_r5.0_8.txt": [16122.550826763],
    "cflp_corn_n100_m100_r5.0_9.txt": [16846.038360034],
    "cflp_corn_n100_m100_r5.0_10.txt": [18264.241302152],
    "cflp_corn_n100_m100_r5.0_11.txt": [17769.328224917],
    "cflp_corn_n100_m100_r5.0_12.txt": [18496.120911648],
    "cflp_corn_n100_m100_r5.0_13.txt": [18670.213542081],
    "cflp_corn_n100_m100_r5.0_14.txt": [18890.400869194],
    "cflp_corn_n100_m100_r5.0_15.txt": [17787.11820802],
    "cflp_corn_n100_m100_r5.0_16.txt": [18594.255562191],
    "cflp_corn_n100_m100_r5.0_17.txt": [17536.698901833],
    "cflp_corn_n100_m100_r5.0_18.txt": [17840.680486713],
    "cflp_corn_n100_m100_r5.0_19.txt": [16677.981827268],
    "cflp_corn_n100_m100_r5.0_20.txt": [18174.252379047],
    # Easy test instances (1000 x 1000)
    "i1000_1.plc": [49509.816283],
    "i1000_2.plc": [50688.099361],
    "i1000_3.plc": [47202.64124],
    "i1000_4.plc": [48868.545165],
    "i1000_5.plc": [50743.542247],
    "i1000_6.plc": [27823.848194],
    "i1000_7.plc": [27252.327368],
    "i1000_8.plc": [27375.377404],
    "i1000_9.plc": [26857.093992],
    "i1000_10.plc": [27186.996215],
    "i1000_11.plc": [22180.338324],
    "i1000_12.plc": [22160.396492],
    "i1000_13.plc": [22648.245244],
    "i1000_14.plc": [22312.017885],
    "i1000_15.plc": [22627.627082],
    "i1000_16.plc": [21331.816412],
    "i1000_17.plc": [21188.891031],
    "i1000_18.plc": [20713.433821],
    "i1000_19.plc": [20537.451973],
    "i1000_20.plc": [21560.863859],
    # Hard test instances (2000 x 2000)
    "p2000-2000-31.plc": [1929201.669659948],
    "p2000-2000-32.plc": [1953642.449903901],
    "p2000-2000-33.plc": [1918045.972964212],
    "p2000-2000-34.plc": [1916455.670809467],
    "p2000-2000-35.plc": [1899636.376243865],
    "p2000-2000-36.plc": [1139219.595105013],
    "p2000-2000-37.plc": [1136995.540252458],
    "p2000-2000-38.plc": [1149261.691855482],
    "p2000-2000-39.plc": [1153261.587371967],
    "p2000-2000-40.plc": [1154591.397009221],
    "p2000-2000-41.plc": [751876.874001226],
    "p2000-2000-42.plc": [749780.77133064],
    "p2000-2000-43.plc": [763162.335598751],
    "p2000-2000-44.plc": [787097.341066275],
    "p2000-2000-45.plc": [762175.878180943],
    "p2000-2000-46.plc": [281246.845669752],
    "p2000-2000-47.plc": [272707.740258233],
    "p2000-2000-48.plc": [276216.104935007],
    "p2000-2000-49.plc": [274280.626885327],
    "p2000-2000-50.plc": [274298.079036553],
    "p2000-2000-51.plc": [194138.93227269],
    "p2000-2000-52.plc": [194518.3],
    "p2000-2000-53.plc": [194329.6],
    "p2000-2000-54.plc": [198441.2],
    "p2000-2000-55.plc": [196469.246795541],
    "p2000-2000-56.plc": [161350.609996602],
    "p2000-2000-57.plc": [157319.490628803],
    "p2000-2000-58.plc": [158310.511106569],
    "p2000-2000-59.plc": [158712.647978297],
    "p2000-2000-60.plc": [155528.26183311],
}


def norm_score_for_instance(filename, orig_idx, score):
    """Return the normalized score for a single instance using FrontierCO's primal gap metric.

    Score is 1.0 for optimal, decreasing toward 0 for worse solutions.
    Formula: 1 - |score - optimal| / max(score, optimal)
    """
    optimal_list = OPTIMAL_SCORES.get(filename)
    if optimal_list is None or not isinstance(score, (int, float)):
        return None
    if orig_idx >= len(optimal_list):
        return None
    optimal = optimal_list[orig_idx]
    denominator = max(score, optimal)
    if denominator == 0:
        return 1.0
    return 1.0 - abs(score - optimal) / denominator




@ray.remote(num_cpus=0)
def _solve_and_eval(main_py_path, instance_id, instance):
    """Ray remote task: load candidate module and run solve + eval for a single instance."""
    try:
        candidate = load_module_from_path(main_py_path)
        solution = candidate.solve(
            instance_id=instance_id,
            n=instance["n"],
            m=instance["m"],
            capacities=instance["capacities"],
            fixed_cost=instance["fixed_cost"],
            demands=instance["demands"],
            trans_costs=instance["trans_costs"],
        )
        score = eval_func(
            n=instance["n"],
            m=instance["m"],
            capacities=instance["capacities"],
            fixed_cost=instance["fixed_cost"],
            demands=instance["demands"],
            trans_costs=instance["trans_costs"],
            total_cost=solution["total_cost"],
            facilities_open=solution["facilities_open"],
            assignments=solution["assignments"],
        )
        return {"score": score, "error": None}
    except Exception as e:
        return {"score": str(e), "error": str(e)}


PROJECT_DIR = Path(__file__).resolve().parent
DEV_DATA_DIR = PROJECT_DIR / "data"
TEST_DATA_DIR = PROJECT_DIR.parent.parent / "downloads" / "FrontierCO__CFLP_test_data"


def evaluate_candidate(main_py_path: str | Path, test: bool = False, timeout: float = 60.0) -> dict:
    """Evaluate a candidate main.py against CFLP instances.

    By default, evaluates on dev instances only (data/, validation 100x100).
    If test=True, evaluates on test instances (downloads/FrontierCO__CFLP_test_data/,
    easy 1000x1000 + hard 2000x2000).

    Each instance is evaluated in parallel via Ray remote tasks.
    Instances that do not complete within *timeout* seconds are marked as timed out.
    """
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)

    main_py_path = str(Path(main_py_path).resolve())
    data_dir = TEST_DATA_DIR if test else DEV_DATA_DIR

    # Gather all data files that have optimal scores
    all_data_files = sorted(data_dir.glob("*.txt")) + sorted(data_dir.glob("*.plc"))
    data_files = [f for f in all_data_files if f.name in OPTIMAL_SCORES]

    # Collect all tasks: map future -> (instance_id, filename, orig_idx)
    future_to_info = {}

    for data_file in data_files:
        filename = data_file.name
        all_instances = load_data(str(data_file))

        for orig_idx, instance in enumerate(all_instances):
            instance_id = f"{Path(filename).stem}_{orig_idx}"
            future = _solve_and_eval.remote(main_py_path, instance_id, instance)
            future_to_info[future] = (instance_id, filename, orig_idx)

    # Wait for results with timeout
    pending = list(future_to_info.keys())
    completed_results = {}

    ready, pending = ray.wait(pending, num_returns=len(pending), timeout=timeout)

    for future, result in zip(ready, ray.get(ready)):
        completed_results[future] = result

    for future in pending:
        instance_id, _, _ = future_to_info[future]
        ray.cancel(future, force=True)

    # Process results
    all_normed_scores = []

    for future, (instance_id, filename, orig_idx) in future_to_info.items():
        if future in completed_results:
            result = completed_results[future]
            if result["error"] is not None:
                all_normed_scores.append(0.0)
                print(f"[instance_id={instance_id}], error=true, normalized_score=0.0000, {result['error']}")
            else:
                normed = norm_score_for_instance(filename, orig_idx, result["score"])
                normed = normed if normed is not None else 0.0
                all_normed_scores.append(normed)
                print(f"[instance_id={instance_id}], error=false, normalized_score={normed:.4f}")
        else:
            all_normed_scores.append(0.0)
            print(f"[instance_id={instance_id}], error=true, normalized_score=0.0000, Timeout after {timeout}s")

    combined_score = sum(all_normed_scores) / len(all_normed_scores) if all_normed_scores else 0.0

    return {"correct": True, "error": "", "combined_score": combined_score}


app = typer.Typer()


@app.command()
def main(
    test: bool = typer.Option(False, "--test", help="Evaluate on test instances instead of dev instances"),
    timeout: float = typer.Option(60.0, "--timeout", help="Total timeout in seconds for all instances"),
):
    project_dir = Path(__file__).parent
    result = evaluate_candidate(project_dir / "main.py", test=test, timeout=timeout)

    print(f"Average normalized score: {result['combined_score']:.4f}")

    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    app()
