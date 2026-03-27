"""Evaluator for the Steiner Tree Problem in Graphs."""

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


def load_data(file_path):
    """
    Load data from an STP file.

    Args:
        file_path: Path to the STP file

    Returns:
        List containing a single dictionary with keys:
        - 'n': number of vertices,
        - 'm': number of edges,
        - 'graph_edges': dictionary mapping (min(u,v), max(u,v)) -> cost,
        - 'terminals': list of terminal vertices.
    """
    data = {
        'n': 0,
        'm': 0,
        'graph_edges': {},
        'terminals': []
    }

    current_section = None

    with open(file_path, 'r') as f:
        for line in f:
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()

            if not line:
                continue

            if line.upper().startswith('SECTION'):
                current_section = line.split()[1].upper()
                continue

            if line.upper() == 'END':
                if current_section == 'TERMINALS':
                    break

                current_section = None
                continue

            if line.upper() == 'EOF':
                break

            if current_section == 'GRAPH':
                parts = line.split()
                if not parts:
                    continue

                if parts[0].upper() == 'NODES':
                    data['n'] = int(parts[1])
                elif parts[0].upper() == 'EDGES':
                    data['m'] = int(parts[1])
                elif parts[0].upper() == 'E':
                    if len(parts) >= 4:
                        u, v, cost = int(parts[1]), int(parts[2]), float(parts[3])
                        data['graph_edges'][(min(u, v), max(u, v))] = cost
                elif parts[0].upper() in ['EA', 'ED', 'EC']:
                    continue

            elif current_section == 'TERMINALS':
                parts = line.split()
                if not parts:
                    continue

                if parts[0].upper() == 'TERMINALS':
                    continue

                if parts[0].upper() == 'T':
                    if len(parts) >= 2:
                        data['terminals'].append(int(parts[1]))

    return [data]


def eval_func(n, m, graph_edges, terminals, declared_cost, num_edges, edges):
    """
    Evaluates the solution for a single test case of the Steiner Tree Problem.

    Args:
        n              : (int) Number of vertices in the graph.
        m              : (int) Number of edges in the graph.
        graph_edges    : (dict) Mapping of (min(u,v), max(u,v)) -> cost.
        terminals      : (list) List of terminal vertices.
        declared_cost  : (float) The total cost declared by the solution.
        num_edges      : (int) Number of edges in the solution.
        edges          : (list) List of edges (each as a tuple (u, v)).

    Returns:
        float: The computed total cost if the solution is valid, otherwise raises Exception.
    """
    import math

    if declared_cost is None or num_edges is None or edges is None:
        raise Exception("Error: The solution must contain 'declared_cost', 'num_edges', and 'edges'.")

    if num_edges != len(edges):
        raise Exception("Error: The number of edges declared does not match the number provided.")

    computed_cost = 0.0
    solution_nodes = set()
    for (u, v) in edges:
        key = (min(u, v), max(u, v))
        if key not in graph_edges:
            raise Exception(f"Error: Edge ({u}, {v}) not found in the input graph.")
        computed_cost += graph_edges[key]
        solution_nodes.update([u, v])

    if abs(declared_cost - computed_cost) > 1e-6:
        raise Exception(f"Error: Declared cost ({declared_cost}) does not match computed cost ({computed_cost}).")

    # Check connectivity among terminal vertices using Union-Find.
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx != ry:
            parent[ry] = rx

    for node in solution_nodes:
        parent[node] = node
    for (u, v) in edges:
        union(u, v)

    # All terminals must be in the solution and connected.
    for t in terminals:
        if t not in parent:
            raise Exception(f"Error: Terminal node {t} is not present in the solution.")

    root = find(terminals[0])
    for t in terminals:
        if find(t) != root:
            raise Exception("Error: Not all terminal nodes are connected in the solution.")

    return computed_cost


OPTIMAL_SCORES = {
    'valid_instances/R25K02EFST.stp': [99.037087811],
    'valid_instances/R25K05EFST.stp': [99.491232075],
    'valid_instances/hc10_valid_1p.stp': [59822.0],
    'valid_instances/hc9_valid_1p.stp': [30241.0],
    'valid_instances/R25K03EFST.stp': [99.215720747],
    'valid_instances/hc6_valid_1p.stp': [4017.0],
    'valid_instances/hc6_valid_0p.stp': [4017.0],
    'valid_instances/hc10_valid_0p.stp': [59822.0],
    'valid_instances/hc9_valid_0p.stp': [30241.0],
    'valid_instances/hc7_valid_0p.stp': [7922.0],
    'valid_instances/hc8_valid_0p.stp': [15331.0],
    'valid_instances/R25K01EFST.stp': [98.961213426],
    'valid_instances/hc8_valid_1p.stp': [15331.0],
    'valid_instances/R25K04EFST.stp': [98.943139188],
    'valid_instances/hc7_valid_1p.stp': [7922.0],
    'easy_test_instances/G207a.stp': [2265834.0],
    'easy_test_instances/G103a.stp': [19938744.0],
    'easy_test_instances/G304a.stp': [6721180.0],
    'easy_test_instances/G305a.stp': [40632152.0],
    'easy_test_instances/G206a.stp': [9175622.0],
    'easy_test_instances/G102a.stp': [15187538.0],
    'easy_test_instances/G307a.stp': [51219090.0],
    'easy_test_instances/G204a.stp': [5313548.0],
    'easy_test_instances/G101a.stp': [3492405.0],
    'easy_test_instances/G205a.stp': [24819583.0],
    'easy_test_instances/G306a.stp': [33949874.0],
    'easy_test_instances/G303a.stp': [27941456.0],
    'easy_test_instances/G104a.stp': [26165528.0],
    'easy_test_instances/G105a.stp': [12507877.0],
    'easy_test_instances/G201a.stp': [3484028.0],
    'easy_test_instances/G302a.stp': [13300990.0],
    'easy_test_instances/G203a.stp': [13155210.0],
    'easy_test_instances/G309a.stp': [11256303.0],
    'easy_test_instances/G107a.stp': [7325530.0],
    'easy_test_instances/G301a.stp': [4797441.0],
    'easy_test_instances/G202a.stp': [6849423.0],
    'easy_test_instances/G308a.stp': [4699474.0],
    'easy_test_instances/G106a.stp': [44547208.0],
    'hard_test_instances/cc6-3p.stp': [20355.0],
    'hard_test_instances/cc11-2u.stp': [617.0],
    'hard_test_instances/bipe2u.stp': [54.0],
    'hard_test_instances/hc7p.stp': [7905.0],
    'hard_test_instances/cc3-10u.stp': [127.0],
    'hard_test_instances/hc11p.stp': [120000.0],
    'hard_test_instances/bip42p.stp': [24657.0],
    'hard_test_instances/bip52p.stp': [24611.0],
    'hard_test_instances/hc10p.stp': [60059.0],
    'hard_test_instances/cc3-11u.stp': [154.0],
    'hard_test_instances/bipa2p.stp': [35393.0],
    'hard_test_instances/cc6-2p.stp': [3271.0],
    'hard_test_instances/cc5-3u.stp': [71.0],
    'hard_test_instances/hc6p.stp': [4003.0],
    'hard_test_instances/cc12-2p.stp': [122007.0],
    'hard_test_instances/cc10-2u.stp': [344.0],
    'hard_test_instances/hc8u.stp': [148.0],
    'hard_test_instances/cc7-3p.stp': [57638.0],
    'hard_test_instances/cc3-5p.stp': [3661.0],
    'hard_test_instances/hc12p.stp': [238889.0],
    'hard_test_instances/cc9-2u.stp': [169.0],
    'hard_test_instances/bip62u.stp': [220.0],
    'hard_test_instances/hc9u.stp': [292.0],
    'hard_test_instances/cc3-4p.stp': [2338.0],
    'hard_test_instances/cc3-12u.stp': [187.0],
    'hard_test_instances/hc9p.stp': [30254.0],
    'hard_test_instances/cc3-12p.stp': [18928.0],
    'hard_test_instances/cc3-4u.stp': [23.0],
    'hard_test_instances/bip62p.stp': [22851.0],
    'hard_test_instances/cc9-2p.stp': [17285.0],
    'hard_test_instances/cc7-3u.stp': [556.0],
    'hard_test_instances/hc8p.stp': [15322.0],
    'hard_test_instances/cc10-2p.stp': [35514.0],
    'hard_test_instances/hc12u.stp': [2330.0],
    'hard_test_instances/cc3-5u.stp': [36.0],
    'hard_test_instances/cc6-2u.stp': [32.0],
    'hard_test_instances/cc12-2u.stp': [1185.0],
    'hard_test_instances/hc6u.stp': [39.0],
    'hard_test_instances/cc5-3p.stp': [7299.0],
    'hard_test_instances/bip52u.stp': [234.0],
    'hard_test_instances/bip42u.stp': [236.0],
    'hard_test_instances/bipa2u.stp': [340.0],
    'hard_test_instances/cc3-11p.stp': [15674.0],
    'hard_test_instances/hc10u.stp': [575.0],
    'hard_test_instances/hc11u.stp': [1159.0],
    'hard_test_instances/cc3-10p.stp': [12866.0],
    'hard_test_instances/cc11-2p.stp': [63608.0],
    'hard_test_instances/cc6-3u.stp': [200.0],
    'hard_test_instances/hc7u.stp': [77.0],
    'hard_test_instances/bipe2p.stp': [5616.0],
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
            graph_edges=instance["graph_edges"],
            terminals=instance["terminals"],
        )
        score = eval_func(
            n=instance["n"],
            m=instance["m"],
            graph_edges=instance["graph_edges"],
            terminals=instance["terminals"],
            declared_cost=solution["declared_cost"],
            num_edges=solution["num_edges"],
            edges=solution["edges"],
        )
        return {"score": score, "error": None}
    except Exception as e:
        return {"score": str(e), "error": str(e)}


def evaluate_candidate(main_py_path: str | Path, data_dir: Path, test: bool = False) -> dict:
    """Evaluate a candidate main.py against STP instances.

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
