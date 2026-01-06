import os
import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

import pandas as pd
from sklearn.metrics import confusion_matrix

from rayevolve.core import run_rayevolve_eval


def validate_paths(
    run_output: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    Outputs validation information.

    Args:
        run_output: Dictionary containing results from 'evaluate_all_graphs'.

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """

    msg = "Validation successful."

    if not run_output["valid"]:
        msg = "Invalid paths found in evaluation."

    return run_output["valid"], msg 


def aggregate_all_graphs(
    results: List[Dict[str, Any]], results_dir: str
) -> Dict[str, Any]:
    """
    Aggregates metrics for train_and_classify. Assumes num_runs=1.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    graph_res = results[0] # need to figure out types here

    metrics = {
        "combined_score": graph_res["total_score"],
    }

    return metrics

def no_kwargs(run_index: int) -> Dict[str, Any]:
    """Provides keyword arguments for runs (none needed)."""
    return {}

def main(program_path: str, results_dir: str):
    """Runs the evaluation using rayevolve.eval."""
    os.makedirs(results_dir, exist_ok=True)

    num_experiment_runs = 1

    # Define a nested function to pass results_dir to the aggregator
    def _aggregator_with_context(
        r: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return aggregate_all_graphs(r, results_dir)

    metrics, correct, error_msg = run_rayevolve_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="evaluate_all_graphs",
        num_runs=num_experiment_runs,
        validate_fn=validate_paths,
        aggregate_metrics_fn=_aggregator_with_context,
        get_experiment_kwargs=no_kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="data set fitting evaluator using rayevolve.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'evaluate_all_graphs' function).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json, extra.npz)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)
