"""Evaluator for the Maximum Independent Set (MIS) problem."""

import sys
import uuid
import json
import importlib.util
from pathlib import Path

import networkx as nx
import pathlib
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
    Load test data for MIS problem.

    Args:
        file_path (str or pathlib.Path): Path to the .mis file.

    Returns:
        list[dict]: [{'graph': nx.Graph}]
    """
    file_path = pathlib.Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix != '.mis':
        raise ValueError(f"Expected .mis file, got {file_path.suffix}")

    try:
        G = nx.Graph()

        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # Problem line (p edge NODES EDGES)
            if parts[0] == 'p' and parts[1] == 'edge':
                num_nodes = int(parts[2])
                G.add_nodes_from(range(1, num_nodes + 1))

            # Edge line (e NODE1 NODE2)
            elif parts[0] == 'e':
                node1 = int(parts[1])
                node2 = int(parts[2])
                G.add_edge(node1, node2)

        test_case = {
            'graph': G
        }

        return [test_case]

    except Exception as e:
        raise Exception(f"Error loading graph from {file_path}: {e}")


def eval_func(graph, mis_nodes):
    """
    Evaluate a Maximum Independent Set solution for correctness.

    Args:
        graph (networkx.Graph): The graph that was solved.
        mis_nodes (list): List of nodes claimed to be in the maximum independent set.

    Returns:
        int: The size of the valid independent set, or raises an exception if invalid.
    """
    if not isinstance(mis_nodes, list):
        raise Exception("mis_nodes must be a list")

    node_set = set(graph.nodes())
    for node in mis_nodes:
        if node not in node_set:
            raise Exception(f"Node {node} in solution does not exist in graph")

    if len(mis_nodes) != len(set(mis_nodes)):
        raise Exception("Duplicate nodes in solution")

    actual_size = len(mis_nodes)

    for i in range(len(mis_nodes)):
        for j in range(i + 1, len(mis_nodes)):
            if graph.has_edge(mis_nodes[i], mis_nodes[j]):
                raise Exception(f"Not an independent set: edge exists between {mis_nodes[i]} and {mis_nodes[j]}")

    return actual_size


OPTIMAL_SCORES = {
    'easy_test_instances/C1000.9.mis': [68.0],
    'easy_test_instances/C125.9.mis': [34.0],
    'easy_test_instances/C2000.5.mis': [16.0],
    'easy_test_instances/C2000.9.mis': [80.0],
    'easy_test_instances/C250.9.mis': [44.0],
    'easy_test_instances/C4000.5.mis': [18.0],
    'easy_test_instances/C500.9.mis': [57.0],
    'easy_test_instances/DSJC1000.5.mis': [15.0],
    'easy_test_instances/DSJC500.5.mis': [13.0],
    'easy_test_instances/MANN_a27.mis': [126.0],
    'easy_test_instances/MANN_a45.mis': [345.0],
    'easy_test_instances/MANN_a81.mis': [1100.0],
    'easy_test_instances/brock200_2.mis': [12.0],
    'easy_test_instances/brock200_4.mis': [17.0],
    'easy_test_instances/brock400_2.mis': [29.0],
    'easy_test_instances/brock400_4.mis': [33.0],
    'easy_test_instances/brock800_2.mis': [24.0],
    'easy_test_instances/brock800_4.mis': [26.0],
    'easy_test_instances/gen200_p0.9_44.mis': [44.0],
    'easy_test_instances/gen200_p0.9_55.mis': [55.0],
    'easy_test_instances/gen400_p0.9_55.mis': [55.0],
    'easy_test_instances/gen400_p0.9_65.mis': [65.0],
    'easy_test_instances/gen400_p0.9_75.mis': [75.0],
    'easy_test_instances/hamming10-4.mis': [40.0],
    'easy_test_instances/hamming8-4.mis': [16.0],
    'easy_test_instances/keller4.mis': [11.0],
    'easy_test_instances/keller5.mis': [27.0],
    'easy_test_instances/keller6.mis': [59.0],
    'easy_test_instances/p_hat1500-1.mis': [12.0],
    'easy_test_instances/p_hat1500-2.mis': [65.0],
    'easy_test_instances/p_hat1500-3.mis': [94.0],
    'easy_test_instances/p_hat300-1.mis': [8.0],
    'easy_test_instances/p_hat300-2.mis': [25.0],
    'easy_test_instances/p_hat300-3.mis': [36.0],
    'easy_test_instances/p_hat700-1.mis': [11.0],
    'easy_test_instances/p_hat700-2.mis': [44.0],
    'easy_test_instances/p_hat700-3.mis': [62.0],
    'hard_test_instances/frb100-40.mis': [98.0],
    'hard_test_instances/frb50-23-1.mis': [50.0],
    'hard_test_instances/frb50-23-2.mis': [50.0],
    'hard_test_instances/frb50-23-3.mis': [50.0],
    'hard_test_instances/frb50-23-4.mis': [50.0],
    'hard_test_instances/frb50-23-5.mis': [50.0],
    'hard_test_instances/frb53-24-1.mis': [53.0],
    'hard_test_instances/frb53-24-2.mis': [53.0],
    'hard_test_instances/frb53-24-3.mis': [53.0],
    'hard_test_instances/frb53-24-4.mis': [53.0],
    'hard_test_instances/frb53-24-5.mis': [53.0],
    'hard_test_instances/frb59-26-1.mis': [59.0],
    'hard_test_instances/frb59-26-2.mis': [59.0],
    'hard_test_instances/frb59-26-3.mis': [59.0],
    'hard_test_instances/frb59-26-4.mis': [59.0],
    'hard_test_instances/frb59-26-5.mis': [59.0],
    'valid_instances/RB_800_1200_0.mis': [47.0],
    'valid_instances/RB_800_1200_1.mis': [50.0],
    'valid_instances/RB_800_1200_10.mis': [37.0],
    'valid_instances/RB_800_1200_11.mis': [50.0],
    'valid_instances/RB_800_1200_12.mis': [49.0],
    'valid_instances/RB_800_1200_13.mis': [44.0],
    'valid_instances/RB_800_1200_14.mis': [41.0],
    'valid_instances/RB_800_1200_15.mis': [45.0],
    'valid_instances/RB_800_1200_16.mis': [43.0],
    'valid_instances/RB_800_1200_17.mis': [40.0],
    'valid_instances/RB_800_1200_18.mis': [40.0],
    'valid_instances/RB_800_1200_19.mis': [36.0],
    'valid_instances/RB_800_1200_2.mis': [36.0],
    'valid_instances/RB_800_1200_3.mis': [50.0],
    'valid_instances/RB_800_1200_4.mis': [44.0],
    'valid_instances/RB_800_1200_5.mis': [47.0],
    'valid_instances/RB_800_1200_6.mis': [45.0],
    'valid_instances/RB_800_1200_7.mis': [38.0],
    'valid_instances/RB_800_1200_8.mis': [38.0],
    'valid_instances/RB_800_1200_9.mis': [50.0],
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
            graph=instance["graph"],
        )
        score = eval_func(
            graph=instance["graph"],
            mis_nodes=solution["mis_nodes"],
        )
        return {"score": score, "error": None}
    except Exception as e:
        return {"score": str(e), "error": str(e)}


def evaluate_candidate(main_py_path: str | Path, data_dir: Path, test: bool = False) -> dict:
    """Evaluate a candidate main.py against MIS instances.

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
        try:
            instances = load_data(str(instance_file))
        except Exception as e:
            print(f"[file={instance_file.name}], error loading data: {e}")
            continue

        for idx, instance in enumerate(instances):
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
