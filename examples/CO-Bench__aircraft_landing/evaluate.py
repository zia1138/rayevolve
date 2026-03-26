"""Evaluator for the Aircraft Landing Scheduling Problem."""

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
    Reads the aircraft landing scheduling problem instance(s) from a file.

    The file may contain one or more cases. Each case has the following format:
        Line 1: <num_planes> <freeze_time>
        For each plane (i = 1, ..., num_planes):
            - A line with 6 numbers:
                  appearance_time earliest_landing_time target_landing_time
                  latest_landing_time penalty_cost_early penalty_cost_late
            - One or more subsequent lines containing exactly num_planes separation times.

    Returns:
        A list of dictionaries, where each dictionary contains the keys:
            - "num_planes"  : int
            - "num_runways" : int
            - "freeze_time" : float
            - "planes"      : list of dicts (one per plane)
            - "separation"  : list of lists of floats
    """
    cases = []
    try:
        with open(input_filename, "r") as f:
            all_lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise RuntimeError(f"Error reading input file '{input_filename}': {e}")

    idx = 0
    total_lines = len(all_lines)
    while idx < total_lines:
        try:
            tokens = all_lines[idx].split()
            num_planes = int(tokens[0])
            freeze_time = float(tokens[1])
        except Exception as e:
            raise ValueError(f"Error parsing case header at line {idx + 1}: {e}")
        idx += 1

        planes = []
        separation = []

        for plane_index in range(num_planes):
            if idx >= total_lines:
                raise ValueError(f"Insufficient lines for plane {plane_index + 1} parameters.")
            params_tokens = all_lines[idx].split()
            idx += 1
            if len(params_tokens) < 6:
                raise ValueError(f"Plane {plane_index + 1}: Expected 6 parameters, got {len(params_tokens)}.")
            try:
                appearance = float(params_tokens[0])
                earliest = float(params_tokens[1])
                target = float(params_tokens[2])
                latest = float(params_tokens[3])
                penalty_early = float(params_tokens[4])
                penalty_late = float(params_tokens[5])
            except Exception as e:
                raise ValueError(f"Plane {plane_index + 1}: Error converting parameters: {e}")

            planes.append({
                "appearance": appearance,
                "earliest": earliest,
                "target": target,
                "latest": latest,
                "penalty_early": penalty_early,
                "penalty_late": penalty_late
            })

            sep_tokens = []
            while len(sep_tokens) < num_planes:
                if idx >= total_lines:
                    raise ValueError(f"Not enough lines to read separation times for plane {plane_index + 1}.")
                sep_tokens.extend(all_lines[idx].split())
                idx += 1
            sep_tokens = sep_tokens[:num_planes]
            try:
                sep_times = [float(token) for token in sep_tokens]
            except Exception as e:
                raise ValueError(f"Plane {plane_index + 1}: Error converting separation times: {e}")
            separation.append(sep_times)

        runway = [[1, 2, 3],
                  [1, 2, 3],
                  [1, 2, 3],
                  [1, 2, 3, 4],
                  [1, 2, 3, 4],
                  [1, 2, 3],
                  [1, 2],
                  [1, 2, 3],
                  [1, 2, 3, 4],
                  [1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5],
                  ]
        case_id = int(input_filename.replace('.txt', '').split('airland')[-1]) - 1
        for num_runways in runway[case_id]:
            case_data = {
                "num_planes": num_planes,
                "num_runways": num_runways,
                "freeze_time": freeze_time,
                "planes": planes,
                "separation": separation,
            }
            cases.append(case_data)
    return cases


def eval_func(num_planes, num_runways, planes, separation, schedule):
    """
    Evaluates a proposed aircraft landing schedule.

    Returns:
        The total penalty (a float) if the schedule is feasible.

    Raises:
        ValueError with an informative message if any constraint is violated.
    """

    if not isinstance(schedule, dict) or len(schedule) != num_planes:
        raise ValueError(f"Schedule must be a dict with exactly {num_planes} entries.")

    for plane_id in range(1, num_planes + 1):
        if plane_id not in schedule:
            raise ValueError(f"Plane {plane_id} is missing in the schedule.")
        entry = schedule[plane_id]
        if not isinstance(entry, dict) or "landing_time" not in entry or "runway" not in entry:
            raise ValueError(f"Schedule entry for plane {plane_id} must contain 'landing_time' and 'runway' keys.")
        runway = entry["runway"]
        if not isinstance(runway, int) or runway < 1 or runway > num_runways:
            raise ValueError(
                f"Plane {plane_id} assigned runway {runway} is invalid. Must be between 1 and {num_runways}.")

    for i in range(1, num_planes + 1):
        landing_time = schedule[i]["landing_time"]
        earliest = planes[i - 1]["earliest"]
        latest = planes[i - 1]["latest"]
        if landing_time < earliest or landing_time > latest:
            raise ValueError(
                f"Plane {i}: Landing time {landing_time} is outside the allowed window [{earliest}, {latest}]."
            )

    for i in range(1, num_planes + 1):
        for j in range(1, num_planes + 1):
            if i == j:
                continue
            entry_i = schedule[i]
            entry_j = schedule[j]
            if entry_i["runway"] == entry_j["runway"]:
                L_i = entry_i["landing_time"]
                L_j = entry_j["landing_time"]
                if L_i <= L_j:
                    required_gap = separation[i - 1][j - 1]
                    if (L_j - L_i) < required_gap:
                        raise ValueError(
                            f"Separation violation on runway {entry_i['runway']}: Plane {i} lands at {L_i} and Plane {j} at {L_j} "
                            f"(required gap: {required_gap})."
                        )

    total_penalty = 0.0
    for i in range(1, num_planes + 1):
        landing_time = schedule[i]["landing_time"]
        target = planes[i - 1]["target"]
        if landing_time < target:
            penalty = (target - landing_time) * planes[i - 1]["penalty_early"]
        elif landing_time > target:
            penalty = (landing_time - target) * planes[i - 1]["penalty_late"]
        else:
            penalty = 0.0
        total_penalty += penalty

    return total_penalty


OPTIMAL_SCORES = {
    "airland1.txt": [700, 90, 0],
    "airland2.txt": [1480, 210, 0],
    "airland3.txt": [820, 60, 0],
    "airland4.txt": [2520, 640, 130, 0],
    "airland5.txt": [3100, 650, 170, 0],
    "airland6.txt": [24442, 554, 0],
    "airland7.txt": [1550, 0],
    "airland8.txt": [1950, 135, 0],
    "airland9.txt": [7848.42, 573.25, 88.72, 0.0],
    "airland10.txt": [17726.06, 1372.21, 246.15, 34.22, 0.0],
    "airland11.txt": [19327.45, 1683.75, 333.53, 69.66, 0.0],
    "airland12.txt": [2549.24, 2204.96, 430.5, 2.86, 0.0],
    "airland13.txt": [58392.69, 4897.92, 821.82, 123.3, 0.0],
}


def norm_score_for_instance(filename, orig_idx, score):
    """Return the normalized score for a single instance, or None if it cannot be computed."""
    optimal_list = OPTIMAL_SCORES.get(filename)
    if optimal_list is None or not isinstance(score, (int, float)):
        return None
    optimal = optimal_list[orig_idx]
    if optimal == 0:
        return (optimal + 1) / (score + 1)
    return optimal / score


def get_dev():
    dev = {'airland1.txt': [0], 'airland10.txt': [2, 1], 'airland11.txt': [0, 1], 'airland12.txt': [3, 4],
           'airland13.txt': [0, 3], 'airland2.txt': [2], 'airland3.txt': [2], 'airland4.txt': [1, 3],
           'airland5.txt': [0, 1],
           'airland6.txt': [1], 'airland7.txt': [1], 'airland8.txt': [2], 'airland9.txt': [0, 1]}
    return dev


@ray.remote(num_cpus=0)
def _solve_and_eval(main_py_path, instance_id, instance):
    """Ray remote task: load candidate module and run solve + eval for a single instance."""
    try:
        candidate = load_module_from_path(main_py_path)
        solution = candidate.solve(
            instance_id=instance_id,
            num_planes=instance["num_planes"],
            num_runways=instance["num_runways"],
            freeze_time=instance["freeze_time"],
            planes=instance["planes"],
            separation=instance["separation"],
        )
        score = eval_func(
            num_planes=instance["num_planes"],
            num_runways=instance["num_runways"],
            planes=instance["planes"],
            separation=instance["separation"],
            schedule=solution["schedule"],
        )
        return {"score": score, "error": None}
    except Exception as e:
        return {"score": str(e), "error": str(e)}


def evaluate_candidate(main_py_path: str | Path, data_dir: Path, test: bool = False, timeout: float = 60.0) -> dict:
    """Evaluate a candidate main.py against airland instances.

    By default, evaluates on dev instances only.
    If test=True, evaluates on the non-dev (test) instances instead.

    Each instance is evaluated in parallel via Ray remote tasks.
    Instances that do not complete within *timeout* seconds are marked as timed out.
    """
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)

    main_py_path = str(Path(main_py_path).resolve())
    dev = get_dev()
    data_files = sorted(data_dir.glob("airland*.txt"))

    # Collect all tasks: map future -> (instance_id, filename, orig_idx)
    future_to_info = {}

    for data_file in data_files:
        filename = data_file.name
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
