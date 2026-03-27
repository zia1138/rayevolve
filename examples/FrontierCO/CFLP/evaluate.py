"""Evaluator for the Capacitated Facility Location Problem."""

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


def load_data(input_path):
    """Read Capacitated Facility Location Problem instance from a file.

    New format:
    n, m (n=facilities, m=customers)
    b1, f1 (capacity and fixed cost of facility 1)
    b2, f2
    ...
    bn, fn
    d1, d2, d3, ..., dm (customer demands)
    c11, c12, c13, ..., c1m (allocation costs for facility 1 to all customers)
    c21, c22, c23, ..., c2m
    ...
    cn1, cn2, cn3, ..., cnm
    """
    # Read all numbers from the file
    with open(input_path, 'r') as f:
        content = f.read()
        # Extract all numbers, ignoring whitespace and empty lines
        all_numbers = [num for num in content.split() if num.strip()]

    pos = 0  # Position in the numbers list

    # Parse dimensions: n (facilities), m (customers)
    n = int(all_numbers[pos])
    pos += 1
    m = int(all_numbers[pos])
    pos += 1

    # Parse facility data: capacity, fixed cost
    capacities = []
    fixed_costs = []
    for _ in range(n):
        if pos + 1 < len(all_numbers):
            capacities.append(float(all_numbers[pos]))
            pos += 1
            fixed_costs.append(float(all_numbers[pos]))
            pos += 1

    # Parse customer demands
    demands = []
    for _ in range(m):
        if pos < len(all_numbers):
            demands.append(float(all_numbers[pos]))
            pos += 1

    # Parse transportation costs
    trans_costs = []
    for _ in range(n):
        facility_costs = []
        for _ in range(m):
            if pos < len(all_numbers):
                facility_costs.append(float(all_numbers[pos]))
                pos += 1
        trans_costs.append(facility_costs)

    # Verify that we have the expected amount of data
    expected_numbers = 2 + 2 * n + m + n * m
    if len(all_numbers) < expected_numbers:
        print(f"Warning: File might be incomplete. Expected {expected_numbers} numbers, found {len(all_numbers)}.")

    case = {"n": m, "m": m, "capacities": capacities, "fixed_cost": fixed_costs, "demands": demands,
            'trans_costs': trans_costs}

    return [case]


def eval_func(n, m, capacities, fixed_cost, demands, trans_costs, facilities_open, assignments):
    """
    Evaluates the solution for the Capacitated Facility Location Problem with Splittable Customer Demand,
    using a weighted average cost for each customer.

    For each customer:
      - The sum of allocations across facilities must equal the customer's demand.
      - The assignment cost is computed as the weighted average of the per-unit costs.
      - No positive allocation is allowed for a facility that is closed.

    Additionally, for each facility:
      - The total allocated demand must not exceed its capacity.

    The total cost is computed as:
         (Sum of fixed costs for all open facilities)
       + (Sum over customers of the weighted average assignment cost)

    Input Parameters:
      - n: Number of facilities (int)
      - m: Number of customers (int)
      - capacities: List of capacities for each facility (list of float)
      - fixed_cost: List of fixed costs for each facility (list of float)
      - demands: List of demands for each customer (list of float)
      - trans_costs: List of lists representing transportation costs from facilities to customers
      - facilities_open: List of n integers (0 or 1) indicating whether each facility is closed or open
      - assignments: List of m lists (each of length n) where assignments[j][i] represents the amount of
                     customer j's demand allocated to facility i

    Returns:
      A floating-point number representing the total cost if the solution is feasible.

    Raises:
      Exception: If any constraint is violated.
    """
    computed_total_cost = 0.0

    # Add fixed costs for open facilities.
    for i in range(n):
        if facilities_open[i] == 1:
            computed_total_cost += fixed_cost[i]

    # Evaluate assignment cost for each customer as a weighted average.
    for j in range(m):
        customer_demand = demands[j]
        allocated_amount = sum(assignments[j])
        if abs(allocated_amount - customer_demand) > 1e-6:
            raise Exception(
                f"Customer {j} demand violation: total assigned amount {allocated_amount} does not equal demand {customer_demand}."
            )
        weighted_cost = 0.0
        for i in range(n):
            allocation = assignments[j][i]
            if allocation < 0:
                raise Exception(
                    f"Customer {j} has a negative allocation {allocation} for facility {i}."
                )
            if allocation > 0 and facilities_open[i] != 1:
                raise Exception(
                    f"Customer {j} has allocation {allocation} for facility {i}, which is closed."
                )
            # Compute fraction of the customer's demand supplied from facility i.
            fraction = allocation / customer_demand if customer_demand > 0 else 0.0
            weighted_cost += fraction * trans_costs[i][j]
        # Add the weighted cost (applied once per customer).
        computed_total_cost += weighted_cost

    # Compute total demand allocated to each facility and check capacity constraints.
    assigned_demand = [0.0] * n
    for i in range(n):
        for j in range(m):
            assigned_demand[i] += assignments[j][i]
    for i in range(n):
        if assigned_demand[i] > capacities[i] + 1e-6:
            excess = assigned_demand[i] - capacities[i]
            raise Exception(
                f"Facility {i} exceeds its capacity by {excess} units."
            )

    return computed_total_cost


OPTIMAL_SCORES = {
    'easy_test_instances/i1000_10.plc': [27186.996215],
    'easy_test_instances/i1000_1.plc': [49509.816283],
    'easy_test_instances/i1000_11.plc': [22180.338324],
    'easy_test_instances/i1000_13.plc': [22648.245244],
    'easy_test_instances/i1000_17.plc': [21188.891031],
    'easy_test_instances/i1000_14.plc': [22312.017885],
    'easy_test_instances/i1000_18.plc': [20713.433821],
    'easy_test_instances/i1000_16.plc': [21331.816412],
    'easy_test_instances/i1000_15.plc': [22627.627082],
    'easy_test_instances/i1000_19.plc': [20537.451973],
    'easy_test_instances/i1000_12.plc': [22160.396492],
    'easy_test_instances/i1000_20.plc': [21560.863859],
    'easy_test_instances/i1000_3.plc': [47202.64124],
    'easy_test_instances/i1000_5.plc': [50743.542247],
    'easy_test_instances/i1000_2.plc': [50688.099361],
    'easy_test_instances/i1000_6.plc': [27823.848194],
    'easy_test_instances/i1000_4.plc': [48868.545165],
    'easy_test_instances/i1000_7.plc': [27252.327368],
    'easy_test_instances/i1000_9.plc': [26857.093992],
    'easy_test_instances/i1000_8.plc': [27375.377404],
    'hard_test_instances/p2000-2000-34.plc': [1916455.670809467],
    'hard_test_instances/p2000-2000-39.plc': [1153261.587371967],
    'hard_test_instances/p2000-2000-36.plc': [1139219.595105013],
    'hard_test_instances/p2000-2000-32.plc': [1953642.449903901],
    'hard_test_instances/p2000-2000-33.plc': [1918045.972964212],
    'hard_test_instances/p2000-2000-37.plc': [1136995.540252458],
    'hard_test_instances/p2000-2000-31.plc': [1929201.669659948],
    'hard_test_instances/p2000-2000-38.plc': [1149261.691855482],
    'hard_test_instances/p2000-2000-35.plc': [1899636.376243865],
    'hard_test_instances/p2000-2000-40.plc': [1154591.397009221],
    'hard_test_instances/p2000-2000-41.plc': [751876.874001226],
    'hard_test_instances/p2000-2000-42.plc': [749780.77133064],
    'hard_test_instances/p2000-2000-43.plc': [763162.335598751],
    'hard_test_instances/p2000-2000-44.plc': [787097.341066275],
    'hard_test_instances/p2000-2000-45.plc': [762175.878180943],
    'hard_test_instances/p2000-2000-46.plc': [281246.845669752],
    'hard_test_instances/p2000-2000-47.plc': [272707.740258233],
    'hard_test_instances/p2000-2000-49.plc': [274280.626885327],
    'hard_test_instances/p2000-2000-48.plc': [276216.104935007],
    'hard_test_instances/p2000-2000-50.plc': [274298.079036553],
    'hard_test_instances/p2000-2000-58.plc': [158310.511106569],
    'hard_test_instances/p2000-2000-57.plc': [157319.490628803],
    'hard_test_instances/p2000-2000-51.plc': [194138.93227269],
    'hard_test_instances/p2000-2000-60.plc': [155528.26183311],
    'hard_test_instances/p2000-2000-56.plc': [161350.609996602],
    'hard_test_instances/p2000-2000-59.plc': [158712.647978297],
    'hard_test_instances/p2000-2000-52.plc': [194518.3],
    'hard_test_instances/p2000-2000-53.plc': [195329.6],
    'hard_test_instances/p2000-2000-54.plc': [198441.2],
    'hard_test_instances/p2000-2000-55.plc': [196469.246795541],
    'valid_instances/cflp_corn_n100_m100_r5.0_11.txt': [17769.328224917],
    'valid_instances/cflp_corn_n100_m100_r5.0_18.txt': [17840.680486713],
    'valid_instances/cflp_corn_n100_m100_r5.0_19.txt': [16677.981827268],
    'valid_instances/cflp_corn_n100_m100_r5.0_17.txt': [17536.698901833],
    'valid_instances/cflp_corn_n100_m100_r5.0_1.txt': [17769.059032386],
    'valid_instances/cflp_corn_n100_m100_r5.0_4.txt': [18555.250515996],
    'valid_instances/cflp_corn_n100_m100_r5.0_15.txt': [17787.11820802],
    'valid_instances/cflp_corn_n100_m100_r5.0_2.txt': [17458.209008233],
    'valid_instances/cflp_corn_n100_m100_r5.0_3.txt': [18172.936618284],
    'valid_instances/cflp_corn_n100_m100_r5.0_7.txt': [17305.736933955],
    'valid_instances/cflp_corn_n100_m100_r5.0_10.txt': [18264.241302152],
    'valid_instances/cflp_corn_n100_m100_r5.0_9.txt': [16846.038360034],
    'valid_instances/cflp_corn_n100_m100_r5.0_20.txt': [18174.252379047],
    'valid_instances/cflp_corn_n100_m100_r5.0_6.txt': [17316.344512835],
    'valid_instances/cflp_corn_n100_m100_r5.0_5.txt': [17202.336476137],
    'valid_instances/cflp_corn_n100_m100_r5.0_13.txt': [18670.213542081],
    'valid_instances/cflp_corn_n100_m100_r5.0_16.txt': [18594.255562191],
    'valid_instances/cflp_corn_n100_m100_r5.0_8.txt': [16122.550826763],
    'valid_instances/cflp_corn_n100_m100_r5.0_12.txt': [18496.120911648],
    'valid_instances/cflp_corn_n100_m100_r5.0_14.txt': [18890.400869194],
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
            facilities_open=solution["facilities_open"],
            assignments=solution["assignments"],
        )
        return {"score": score, "error": None}
    except Exception as e:
        return {"score": str(e), "error": str(e)}


def evaluate_candidate(main_py_path: str | Path, data_dir: Path, test: bool = False) -> dict:
    """Evaluate a candidate main.py against CFLP instances.

    By default, evaluates on dev instances (valid_instances/).
    If test=True, evaluates on easy_test_instances/ + hard_test_instances/.
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
