"""Evaluator for CO-Bench Flow Shop Scheduling."""

import sys
import json
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cobench_utils import load_module_from_path, write_results


def load_data(file_path):
    """Load flow shop test cases from a Taillard-format file."""
    test_cases = []
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

            test_cases.append({
                "n": n,
                "m": m,
                "matrix": matrix,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
            })
        else:
            i += 1

    return test_cases


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
    """Evaluate a flow shop solution. Returns (score, error_msg)."""
    n = case["n"]
    m = case["m"]
    matrix = case["matrix"]
    lower_bound = case["lower_bound"]

    job_sequence = solution.get("job_sequence", [])

    if len(job_sequence) != n or set(job_sequence) != set(range(1, n + 1)):
        return 0.0, f"Invalid job_sequence: not a permutation of [1..{n}]"

    makespan = compute_makespan(job_sequence, matrix, n, m)
    score = lower_bound / makespan
    return score, None


INSTANCE_TIMEOUT = 60  # seconds per instance


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Instance timed out")


if __name__ == "__main__":
    data_dir = Path("data")
    data_files = sorted(data_dir.glob("tai*.txt"))

    if not data_files:
        write_results({"correct": False, "combined_score": 0.0, "error": "No data files found in data/"})
        sys.exit(0)

    module = load_module_from_path("main.py")

    all_scores = []
    errors = []

    for fpath in data_files:
        cases = load_data(fpath)
        for case_idx, case in enumerate(cases):
            case_id = f"{fpath.stem}_{case_idx}"
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(INSTANCE_TIMEOUT)
                solution = module.solve(**case)
                signal.alarm(0)  # cancel alarm
                score, err = eval_solution(case, solution)
                if err:
                    errors.append(f"{case_id}: {err}")
                    all_scores.append(0.0)
                else:
                    all_scores.append(score)
            except Exception as e:
                errors.append(f"{case_id}: {e}")
                all_scores.append(0.0)

    combined_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    is_correct = len(errors) == 0

    result = {
        "correct": is_correct,
        "combined_score": combined_score,
        "error": "; ".join(errors[:5]) if errors else None,
        "num_instances": len(all_scores),
        "num_errors": len(errors),
    }
    write_results(result)
