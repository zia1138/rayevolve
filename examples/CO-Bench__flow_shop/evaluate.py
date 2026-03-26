"""Evaluator for the CO-Bench Flow Shop Scheduling Problem.

CO-Bench specifies a 60-second per-instance timeout using SIGALRM. This evaluator
uses a relaxed approach: all dev instances run in parallel as Ray remote tasks with
a single global timeout via ray.wait(), and instances that exceed it are cancelled.
"""

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


def load_data(file_path):
    """
    Load flow shop instances from a Taillard-format benchmark file.

    Each instance in the file has:
        - Header: "number of jobs, number of machines, initial seed, upper bound and lower bound :"
        - A line with: n  m  seed  upper_bound  lower_bound
        - "processing times :" header
        - m lines of n integers each (machine_times[machine][job])

    Returns:
        A list of dicts with keys: n, m, matrix (job x machine), upper_bound, lower_bound.
    """
    cases = []
    with open(file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("number of jobs"):
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            header_tokens = lines[i].strip().split()
            n = int(header_tokens[0])
            m = int(header_tokens[1])
            upper_bound = int(header_tokens[3])
            lower_bound = int(header_tokens[4])
            i += 1

            while i < len(lines) and lines[i].strip() == "":
                i += 1
            if i < len(lines) and lines[i].strip().lower().startswith("processing times"):
                i += 1

            # Read m lines of processing times (each line has n ints, one per machine)
            machine_times = []
            for _ in range(m):
                while i < len(lines) and lines[i].strip() == "":
                    i += 1
                row = [int(t) for t in lines[i].strip().split()]
                machine_times.append(row)
                i += 1

            # Transpose: machine_times[machine][job] -> matrix[job][machine]
            matrix = [[machine_times[machine][job] for machine in range(m)] for job in range(n)]

            cases.append({
                "n": n,
                "m": m,
                "matrix": matrix,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
            })
        else:
            i += 1

    return cases


def compute_makespan(job_sequence, matrix, n, m):
    """Compute makespan using the classical flow shop recurrence."""
    seq = [j - 1 for j in job_sequence]  # convert to 0-indexed
    completion = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            proc = matrix[seq[i]][j]
            if i == 0 and j == 0:
                completion[i][j] = proc
            elif i == 0:
                completion[i][j] = completion[i][j - 1] + proc
            elif j == 0:
                completion[i][j] = completion[i - 1][j] + proc
            else:
                completion[i][j] = max(completion[i - 1][j], completion[i][j - 1]) + proc
    return completion[-1][-1]


def eval_solution(case, solution):
    """Validate solution and compute makespan. Returns (makespan, error_msg)."""
    n = case["n"]
    m = case["m"]
    matrix = case["matrix"]

    job_sequence = solution.get("job_sequence", [])

    if len(job_sequence) != n or set(job_sequence) != set(range(1, n + 1)):
        return None, f"Invalid job_sequence: not a permutation of [1..{n}]"

    makespan = compute_makespan(job_sequence, matrix, n, m)
    return makespan, None


# Lower bounds from Taillard benchmark instances (10 instances per file).
OPTIMAL_SCORES = {
    "tai20_5.txt": [1232, 1290, 1073, 1268, 1198, 1180, 1226, 1170, 1206, 1082],
    "tai20_10.txt": [1448, 1479, 1407, 1308, 1325, 1290, 1388, 1363, 1472, 1356],
    "tai20_20.txt": [1911, 1711, 1844, 1810, 1899, 1875, 1875, 1880, 1840, 1900],
    "tai50_5.txt": [2712, 2808, 2596, 2740, 2837, 2793, 2689, 2667, 2527, 2776],
    "tai50_10.txt": [2907, 2821, 2801, 2968, 2908, 2941, 3062, 2959, 2795, 3046],
    "tai50_20.txt": [3480, 3424, 3351, 3336, 3313, 3460, 3427, 3383, 3457, 3438],
    "tai100_5.txt": [5437, 5208, 5130, 4963, 5195, 5063, 5198, 5038, 5385, 5272],
    "tai100_10.txt": [5759, 5345, 5623, 5732, 5431, 5246, 5523, 5556, 5779, 5830],
    "tai100_20.txt": [5851, 6099, 6099, 6072, 6009, 6144, 5991, 6084, 5979, 6298],
    "tai200_10.txt": [10816, 10422, 10886, 10794, 10437, 10255, 10761, 10663, 10348, 10616],
    "tai200_20.txt": [10979, 10947, 11150, 11127, 11132, 11085, 11194, 11126, 10965, 11122],
    "tai500_20.txt": [25922, 26353, 26320, 26424, 26181, 26401, 26300, 26429, 25891, 26315],
}


def norm_score_for_instance(filename, idx, makespan):
    """Return the normalized score for a single instance, or None if it cannot be computed."""
    optimal_list = OPTIMAL_SCORES.get(filename)
    if optimal_list is None or not isinstance(makespan, (int, float)):
        return None
    if idx >= len(optimal_list):
        return None
    if makespan == 0:
        return None
    lower_bound = optimal_list[idx]
    return lower_bound / makespan


def get_dev():
    """Return dev instance indices per data file (3 of 10 per file)."""
    return {
        "tai20_5.txt": [0, 3, 7],
        "tai20_10.txt": [0, 3, 7],
        "tai20_20.txt": [0, 3, 7],
        "tai50_5.txt": [0, 3, 7],
        "tai50_10.txt": [0, 3, 7],
        "tai50_20.txt": [0, 3, 7],
        "tai100_5.txt": [0, 3, 7],
        "tai100_10.txt": [0, 3, 7],
        "tai100_20.txt": [0, 3, 7],
        "tai200_10.txt": [0, 3, 7],
        "tai200_20.txt": [0, 3, 7],
        "tai500_20.txt": [0, 3, 7],
    }


@ray.remote(num_cpus=0)
def _solve_and_eval(main_py_path, instance_id, instance):
    """Ray remote task: load candidate module and run solve + eval for a single instance."""
    try:
        candidate = load_module_from_path(main_py_path)
        solution = candidate.solve(
            instance_id=instance_id,
            n=instance["n"],
            m=instance["m"],
            matrix=instance["matrix"],
            upper_bound=instance["upper_bound"],
            lower_bound=instance["lower_bound"],
        )
        makespan, err = eval_solution(instance, solution)
        if err:
            return {"score": None, "error": err}
        return {"score": makespan, "error": None}
    except Exception as e:
        return {"score": str(e), "error": str(e)}


def evaluate_candidate(main_py_path: str | Path, data_dir: Path, test: bool = False, timeout: float = 60.0) -> dict:
    """Evaluate a candidate main.py against Taillard flow shop instances.

    By default, evaluates on dev instances only.
    If test=True, evaluates on the non-dev (test) instances instead.

    Each instance is evaluated in parallel via Ray remote tasks.
    Instances that do not complete within *timeout* seconds are marked as timed out.
    """
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)

    main_py_path = str(Path(main_py_path).resolve())
    dev = get_dev()
    data_files = sorted(data_dir.glob("tai*.txt"))

    # Collect all tasks: map future -> (instance_id, filename, orig_idx)
    future_to_info = {}

    for data_file in data_files:
        filename = data_file.name
        if filename not in OPTIMAL_SCORES:
            continue
        all_instances = load_data(str(data_file))
        dev_indices = set(dev.get(filename, []))

        if test:
            selected = [(i, inst) for i, inst in enumerate(all_instances) if i not in dev_indices]
        else:
            selected = [(i, inst) for i, inst in enumerate(all_instances) if i in dev_indices]

        for orig_idx, instance in selected:
            instance_id = f"{filename.replace('.txt', '')}_{orig_idx}"
            future = _solve_and_eval.remote(main_py_path, instance_id, instance)
            future_to_info[future] = (instance_id, filename, orig_idx)

    # Wait for results with timeout
    pending = list(future_to_info.keys())
    completed_results = {}  # future -> result dict

    ready, pending = ray.wait(pending, num_returns=len(pending), timeout=timeout)

    # Fetch completed results
    for future, result in zip(ready, ray.get(ready)):
        completed_results[future] = result

    # Report timed-out instances and cancel them
    for future in pending:
        instance_id, _, _ = future_to_info[future]
        ray.cancel(future, force=True)

    # Process results: one line per instance_id
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
    data_dir = project_dir / "data"
    result = evaluate_candidate(project_dir / "main.py", data_dir, test=test, timeout=timeout)

    print(f"Average normalized score: {result['combined_score']:.4f}")

    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    app()
