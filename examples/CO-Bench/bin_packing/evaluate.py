"""Evaluator for CO-Bench 1D Bin Packing."""

import sys
import json
from pathlib import Path

# Add parent dir so we can import cobench_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cobench_utils import load_module_from_path, write_results


def load_data(input_file_path):
    """Load bin packing test cases from a CO-Bench data file."""
    cases = []
    with open(input_file_path, "r") as fin:
        in_lines = [line.strip() for line in fin if line.strip()]

    num_cases = int(in_lines[0])
    pos = 1
    for _ in range(num_cases):
        prob_id = in_lines[pos]
        pos += 1
        header_parts = in_lines[pos].split()
        pos += 1
        bin_capacity = float(header_parts[0])
        num_items = int(header_parts[1])
        best_known = int(header_parts[2])

        items = []
        for _ in range(num_items):
            items.append(float(in_lines[pos]))
            pos += 1

        cases.append({
            "id": prob_id,
            "bin_capacity": bin_capacity,
            "num_items": num_items,
            "best_known": best_known,
            "items": items,
        })
    return cases


def eval_solution(case, solution):
    """Evaluate a bin packing solution. Returns (score, error_msg)."""
    num_items = case["num_items"]
    bin_capacity = case["bin_capacity"]
    best_known = case["best_known"]
    items = case["items"]

    num_bins = solution.get("num_bins", 0)
    bins = solution.get("bins", [])

    if len(bins) != num_bins:
        return 0.0, "num_bins does not match len(bins)"

    item_counts = [0] * (num_items + 1)
    for bin_idx, bin_items in enumerate(bins, start=1):
        bin_total = 0
        for item_idx in bin_items:
            if item_idx < 1 or item_idx > num_items:
                return 0.0, f"Bin {bin_idx}: invalid item index {item_idx}"
            bin_total += items[item_idx - 1]
            item_counts[item_idx] += 1
        if bin_total > bin_capacity + 1e-9:
            return 0.0, f"Bin {bin_idx}: exceeds capacity ({bin_total} > {bin_capacity})"

    for i in range(1, num_items + 1):
        if item_counts[i] != 1:
            return 0.0, f"Item {i} appears {item_counts[i]} times"

    score = best_known / num_bins
    return score, None


if __name__ == "__main__":
    data_dir = Path("data")
    data_files = sorted(data_dir.glob("binpack*.txt"))

    if not data_files:
        write_results({"correct": False, "combined_score": 0.0, "error": "No data files found in data/"})
        sys.exit(0)

    module = load_module_from_path("main.py")

    all_scores = []
    errors = []

    for fpath in data_files:
        cases = load_data(fpath)
        for case in cases:
            try:
                solution = module.solve(**{k: v for k, v in case.items() if k != "best_known"})
                score, err = eval_solution(case, solution)
                if err:
                    errors.append(f"{case['id']}: {err}")
                    all_scores.append(0.0)
                else:
                    all_scores.append(score)
            except Exception as e:
                errors.append(f"{case['id']}: {e}")
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
