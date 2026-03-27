"""Evaluator for the Minimum Dominant Set (MDS) problem."""

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
    Load test data for an MDS instance.

    Args:
        file_path (str or pathlib.Path): Path to the .gr file.

    Returns:
        list[dict]: [{'graph': nx.Graph}]
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    G = nx.Graph()
    edges = []

    with file_path.open('r') as f:
        for line in f:
            if not line or line[0].isspace():
                continue

            if line[0] == 'p':
                _, fmt, n_nodes, *_ = line.split()
                if fmt != 'ds':
                    raise ValueError(f"Unexpected format: {fmt}")
                G.add_nodes_from(range(1, int(n_nodes) + 1))
                continue

            # Otherwise it must be an edge line: "u v"
            u_str, v_str = line.split()
            edges.append((int(u_str), int(v_str)))

    G.add_edges_from(edges)

    return [{'graph': G}]


def eval_func(graph, mds_nodes):
    """
    Evaluate a Minimum Dominant Set solution for correctness.

    Args:
        graph (networkx.Graph): The graph that was solved.
        mds_nodes (list): List of nodes claimed to be in the minimum dominant set.

    Returns:
        int: The size of the valid dominant set, or raises an exception if invalid.
    """
    if not isinstance(mds_nodes, list):
        raise Exception("mds_nodes must be a list")

    node_set = set(graph.nodes())
    for node in mds_nodes:
        if node not in node_set:
            raise Exception(f"Node {node} in solution does not exist in graph")

    if len(mds_nodes) != len(set(mds_nodes)):
        raise Exception("Duplicate nodes in solution")

    actual_size = len(mds_nodes)

    dominated_nodes = set(mds_nodes)
    for node in mds_nodes:
        dominated_nodes.update(graph.neighbors(node))

    if dominated_nodes != node_set:
        undominated = node_set - dominated_nodes
        raise Exception(f"Not a dominant set: nodes {undominated} are not dominated")

    return actual_size


OPTIMAL_SCORES = {
    'easy_test_instances/exact_066.gr': [707.0],
    'easy_test_instances/exact_088.gr': [707.0],
    'easy_test_instances/exact_075.gr': [706.0],
    'easy_test_instances/exact_093.gr': [706.0],
    'easy_test_instances/exact_097.gr': [706.0],
    'easy_test_instances/exact_081.gr': [1216.0],
    'easy_test_instances/exact_057.gr': [705.0],
    'easy_test_instances/exact_063.gr': [805.0],
    'easy_test_instances/exact_072.gr': [805.0],
    'easy_test_instances/exact_092.gr': [1183.0],
    'easy_test_instances/exact_069.gr': [1171.0],
    'easy_test_instances/exact_033.gr': [5539.0],
    'easy_test_instances/exact_071.gr': [2689.0],
    'easy_test_instances/exact_051.gr': [849.0],
    'easy_test_instances/exact_067.gr': [989.0],
    'easy_test_instances/exact_076.gr': [1597.0],
    'easy_test_instances/exact_058.gr': [740.0],
    'easy_test_instances/exact_056.gr': [1512.0],
    'easy_test_instances/exact_083.gr': [1866.0],
    'easy_test_instances/exact_034.gr': [5842.0],
    'hard_test_instances/heuristic_049.gr': [3062.0],
    'hard_test_instances/heuristic_065.gr': [3159.0],
    'hard_test_instances/heuristic_016.gr': [3352.0],
    'hard_test_instances/heuristic_042.gr': [2999.0],
    'hard_test_instances/heuristic_017.gr': [3330.0],
    'hard_test_instances/heuristic_019.gr': [3062.0],
    'hard_test_instances/heuristic_036.gr': [3050.0],
    'hard_test_instances/heuristic_067.gr': [3277.0],
    'hard_test_instances/heuristic_097.gr': [3025.0],
    'hard_test_instances/heuristic_015.gr': [3077.0],
    'hard_test_instances/heuristic_059.gr': [2997.0],
    'hard_test_instances/heuristic_037.gr': [3054.0],
    'hard_test_instances/heuristic_026.gr': [3025.0],
    'hard_test_instances/heuristic_060.gr': [3001.0],
    'hard_test_instances/heuristic_078.gr': [2829.0],
    'hard_test_instances/heuristic_044.gr': [2937.0],
    'hard_test_instances/heuristic_003.gr': [637607.0],
    'hard_test_instances/heuristic_066.gr': [1047.0],
    'hard_test_instances/heuristic_074.gr': [331531.0],
    'hard_test_instances/heuristic_077.gr': [427644.0],
    'valid_instances/ba_graph_large_train_12.txt': [96.0],
    'valid_instances/ba_graph_large_train_11.txt': [93.0],
    'valid_instances/ba_graph_large_train_10.txt': [123.0],
    'valid_instances/ba_graph_large_train_19.txt': [116.0],
    'valid_instances/ba_graph_large_train_14.txt': [93.0],
    'valid_instances/ba_graph_large_train_0.txt': [118.0],
    'valid_instances/ba_graph_large_train_17.txt': [106.0],
    'valid_instances/ba_graph_large_train_6.txt': [107.0],
    'valid_instances/ba_graph_large_train_18.txt': [117.0],
    'valid_instances/ba_graph_large_train_13.txt': [120.0],
    'valid_instances/ba_graph_large_train_7.txt': [86.0],
    'valid_instances/ba_graph_large_train_5.txt': [114.0],
    'valid_instances/ba_graph_large_train_3.txt': [118.0],
    'valid_instances/ba_graph_large_train_9.txt': [114.0],
    'valid_instances/ba_graph_large_train_15.txt': [92.0],
    'valid_instances/ba_graph_large_train_16.txt': [112.0],
    'valid_instances/ba_graph_large_train_8.txt': [124.0],
    'valid_instances/ba_graph_large_train_2.txt': [116.0],
    'valid_instances/ba_graph_large_train_4.txt': [121.0],
    'valid_instances/ba_graph_large_train_1.txt': [124.0],
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
            mds_nodes=solution["mds_nodes"],
        )
        return {"score": score, "error": None}
    except Exception as e:
        return {"score": str(e), "error": str(e)}


def evaluate_candidate(main_py_path: str | Path, data_dir: Path, test: bool = False) -> dict:
    """Evaluate a candidate main.py against MDS instances.

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
