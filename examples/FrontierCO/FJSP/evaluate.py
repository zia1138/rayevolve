"""Evaluator for the Flexible Job Shop Scheduling Problem (FJSP)."""

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


def load_data(filename):
    """Read Flexible Job Shop Scheduling Problem instance from a file.

    Format:
    <number of jobs> <number of machines>
    <number of operations for job 1> <number of machines for op 1> <machine 1> <time 1> <machine 2> <time 2> ... <number of machines for op 2> <machine 1> <time 1> ...
    <number of operations for job 2> ...
    ...
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Parse first line: number of jobs and machines
    parts = lines[0].split()
    num_jobs = int(parts[0])
    num_machines = int(parts[1])

    # Parse job information
    jobs = []
    for i in range(1, num_jobs + 1):
        if i < len(lines):
            job_data = list(map(int, lines[i].split()))
            job_operations = []

            # Parse operations for this job
            idx = 1  # Skip the first number (number of operations)
            num_operations = job_data[0]

            for _ in range(num_operations):
                if idx < len(job_data):
                    num_machines_for_op = job_data[idx]
                    idx += 1

                    # Parse machine-time pairs for this operation
                    machine_time_pairs = []
                    for _ in range(num_machines_for_op):
                        if idx + 1 < len(job_data):
                            machine = job_data[idx]
                            time = job_data[idx + 1]
                            machine_time_pairs.append((machine, time))
                            idx += 2

                    job_operations.append(machine_time_pairs)

            jobs.append(job_operations)

    # Validate that we have the expected amount of data
    if len(jobs) != num_jobs:
        print(f"Warning: Expected {num_jobs} jobs, found {len(jobs)}.")

    case = {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "jobs": jobs
    }

    return [case]


def eval_func(num_jobs, num_machines, jobs, machine_assignments, start_times):
    """
    Evaluates the solution for the Flexible Job Shop Scheduling Problem.

    Args:
        num_jobs (int): Number of jobs.
        num_machines (int): Number of machines.
        jobs (list): A list of jobs, where each job is a list of operations.
        machine_assignments (list): A list of machine assignments for each operation.
        start_times (list): A list of start times for each operation.

    Returns:
        float: The makespan if the solution is feasible.

    Raises:
        Exception: If any constraint is violated.
    """
    # Flatten job operations for indexing
    flat_operations = []
    for job in jobs:
        for operation in job:
            flat_operations.append(operation)

    # Validate machine assignments
    for i, (operation, assigned_machine) in enumerate(zip(flat_operations, machine_assignments)):
        # Check if assigned machine is compatible with operation
        compatible_machines = [machine for machine, _ in operation]
        if assigned_machine not in compatible_machines:
            raise Exception(f"Operation {i} assigned to incompatible machine {assigned_machine}")

    # Track job precedence constraints
    job_op_end_times = {}  # (job_idx, op_idx_within_job) -> end_time
    op_idx = 0

    # Calculate end times and check precedence constraints
    for job_idx, job in enumerate(jobs):
        for op_idx_within_job, operation in enumerate(job):
            current_op_idx = op_idx

            # Get assigned machine and processing time
            assigned_machine = machine_assignments[current_op_idx]
            processing_time = next(time for machine, time in operation if machine == assigned_machine)

            start_time = start_times[current_op_idx]
            end_time = start_time + processing_time

            # Check job precedence constraint
            if op_idx_within_job > 0:
                prev_end_time = job_op_end_times.get((job_idx, op_idx_within_job - 1), 0)
                if start_time < prev_end_time:
                    raise Exception(f"Operation {current_op_idx} starts at {start_time}, "
                                    f"before previous operation in job {job_idx} ends at {prev_end_time}")

            job_op_end_times[(job_idx, op_idx_within_job)] = end_time
            op_idx += 1

    # Check machine capacity constraints (no overlap on same machine)
    machine_schedules = {}  # machine -> list of (start_time, end_time) tuples
    op_idx = 0

    for job in jobs:
        for operation in job:
            assigned_machine = machine_assignments[op_idx]
            processing_time = next(time for machine, time in operation if machine == assigned_machine)

            start_time = start_times[op_idx]
            end_time = start_time + processing_time

            if assigned_machine not in machine_schedules:
                machine_schedules[assigned_machine] = []

            # Check for overlaps on this machine
            for other_start, other_end in machine_schedules[assigned_machine]:
                if not (end_time <= other_start or start_time >= other_end):
                    raise Exception(f"Operation at time {start_time}-{end_time} overlaps with another "
                                    f"operation on machine {assigned_machine} at {other_start}-{other_end}")

            machine_schedules[assigned_machine].append((start_time, end_time))
            op_idx += 1

    # Calculate makespan
    makespan = max(end_time for machine_times in machine_schedules.values()
                   for _, end_time in machine_times)

    return makespan


OPTIMAL_SCORES = {
    'easy_test_instances/Behnke1.fjs': [90.0],
    'easy_test_instances/Behnke10.fjs': [127.0],
    'easy_test_instances/Behnke11.fjs': [231.0],
    'easy_test_instances/Behnke12.fjs': [220.0],
    'easy_test_instances/Behnke13.fjs': [231.0],
    'easy_test_instances/Behnke14.fjs': [232.0],
    'easy_test_instances/Behnke15.fjs': [227.0],
    'easy_test_instances/Behnke16.fjs': [417.0],
    'easy_test_instances/Behnke17.fjs': [406.0],
    'easy_test_instances/Behnke18.fjs': [404.0],
    'easy_test_instances/Behnke19.fjs': [407.0],
    'easy_test_instances/Behnke2.fjs': [91.0],
    'easy_test_instances/Behnke20.fjs': [404.0],
    'easy_test_instances/Behnke21.fjs': [85.0],
    'easy_test_instances/Behnke22.fjs': [87.0],
    'easy_test_instances/Behnke23.fjs': [85.0],
    'easy_test_instances/Behnke24.fjs': [87.0],
    'easy_test_instances/Behnke25.fjs': [87.0],
    'easy_test_instances/Behnke26.fjs': [113.0],
    'easy_test_instances/Behnke27.fjs': [122.0],
    'easy_test_instances/Behnke28.fjs': [114.0],
    'easy_test_instances/Behnke29.fjs': [118.0],
    'easy_test_instances/Behnke3.fjs': [91.0],
    'easy_test_instances/Behnke30.fjs': [121.0],
    'easy_test_instances/Behnke31.fjs': [226.0],
    'easy_test_instances/Behnke32.fjs': [222.0],
    'easy_test_instances/Behnke33.fjs': [226.0],
    'easy_test_instances/Behnke34.fjs': [221.0],
    'easy_test_instances/Behnke35.fjs': [214.0],
    'easy_test_instances/Behnke36.fjs': [392.0],
    'easy_test_instances/Behnke37.fjs': [399.0],
    'easy_test_instances/Behnke38.fjs': [395.0],
    'easy_test_instances/Behnke39.fjs': [393.0],
    'easy_test_instances/Behnke4.fjs': [97.0],
    'easy_test_instances/Behnke40.fjs': [421.0],
    'easy_test_instances/Behnke41.fjs': [87.0],
    'easy_test_instances/Behnke42.fjs': [87.0],
    'easy_test_instances/Behnke43.fjs': [86.0],
    'easy_test_instances/Behnke44.fjs': [84.0],
    'easy_test_instances/Behnke45.fjs': [87.0],
    'easy_test_instances/Behnke46.fjs': [115.0],
    'easy_test_instances/Behnke47.fjs': [117.0],
    'easy_test_instances/Behnke48.fjs': [125.0],
    'easy_test_instances/Behnke49.fjs': [113.0],
    'easy_test_instances/Behnke5.fjs': [91.0],
    'easy_test_instances/Behnke50.fjs': [124.0],
    'easy_test_instances/Behnke51.fjs': [220.0],
    'easy_test_instances/Behnke52.fjs': [215.0],
    'easy_test_instances/Behnke53.fjs': [213.0],
    'easy_test_instances/Behnke54.fjs': [225.0],
    'easy_test_instances/Behnke55.fjs': [222.0],
    'easy_test_instances/Behnke56.fjs': [394.0],
    'easy_test_instances/Behnke57.fjs': [393.0],
    'easy_test_instances/Behnke58.fjs': [406.0],
    'easy_test_instances/Behnke59.fjs': [404.0],
    'easy_test_instances/Behnke6.fjs': [125.0],
    'easy_test_instances/Behnke60.fjs': [402.0],
    'easy_test_instances/Behnke7.fjs': [125.0],
    'easy_test_instances/Behnke8.fjs': [124.0],
    'easy_test_instances/Behnke9.fjs': [127.0],
    'hard_test_instances/73.txt': [3723.0],
    'hard_test_instances/74.txt': [3706.0],
    'hard_test_instances/75.txt': [3436.0],
    'hard_test_instances/76.txt': [3790.0],
    'hard_test_instances/77.txt': [7406.0],
    'hard_test_instances/78.txt': [7570.0],
    'hard_test_instances/79.txt': [7040.0],
    'hard_test_instances/80.txt': [7825.0],
    'hard_test_instances/81.txt': [2276.0],
    'hard_test_instances/82.txt': [2520.0],
    'hard_test_instances/83.txt': [2290.0],
    'hard_test_instances/84.txt': [2581.0],
    'hard_test_instances/85.txt': [4901.0],
    'hard_test_instances/86.txt': [5109.0],
    'hard_test_instances/87.txt': [4954.0],
    'hard_test_instances/88.txt': [4994.0],
    'hard_test_instances/89.txt': [1810.0],
    'hard_test_instances/90.txt': [1778.0],
    'hard_test_instances/91.txt': [1707.0],
    'hard_test_instances/92.txt': [1923.0],
    'hard_test_instances/93.txt': [3553.0],
    'hard_test_instances/94.txt': [3790.0],
    'hard_test_instances/95.txt': [3586.0],
    'hard_test_instances/96.txt': [3896.0],
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
            num_jobs=instance["num_jobs"],
            num_machines=instance["num_machines"],
            jobs=instance["jobs"],
        )
        score = eval_func(
            num_jobs=instance["num_jobs"],
            num_machines=instance["num_machines"],
            jobs=instance["jobs"],
            machine_assignments=solution["machine_assignments"],
            start_times=solution["start_times"],
        )
        return {"score": score, "error": None}
    except Exception as e:
        return {"score": str(e), "error": str(e)}


def evaluate_candidate(main_py_path: str | Path, data_dir: Path, test: bool = False) -> dict:
    """Evaluate a candidate main.py against FJSP instances.

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
