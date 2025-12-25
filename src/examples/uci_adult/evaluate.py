"""
Evaluator for circle packing example (n=26) with improved timeout handling
"""

import os
import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

import pandas as pd
from sklearn.metrics import roc_auc_score

from examples.uci_adult.initial import X_val_2
from rayevolve.core import run_rayevolve_eval


def validate_prediction(
    run_output: Tuple[pd.DataFrame, pd.DataFrame],
) -> Tuple[bool, Optional[str]]:
    """
    Validates classification results from 'train_and_classify'.

    Args:
        run_output: Tuple (X_val, y_val) from train_and_classify.

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    X_val, y_val = run_output
    msg = "X_val shape: {}, y_val shape: {}".format(X_val.shape, y_val.shape)
    
    return True, msg


def aggregate_train_and_classify(
    results: List[Tuple[pd.DataFrame, pd.DataFrame]], results_dir: str
) -> Dict[str, Any]:
    """
    Aggregates metrics for train_and_classify. Assumes num_runs=1.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    X_val_2, y_val2_proba = results[0]

    y_true_df = pd.read_csv("y_val.csv", index_col=0)
    y_true_df = y_true_df.loc[X_val_2.index]

    # With binary numeric labels, compute AUC directly from probabilities
    y_true = y_true_df.iloc[:, 0] if isinstance(y_true_df, pd.DataFrame) else y_true_df
    y_score = y_val2_proba.iloc[:, 0]
    auc = roc_auc_score(y_true, y_score)

    metrics = {
        "combined_score": float(auc),
    }

    return metrics


def main(program_path: str, results_dir: str):
    """Runs the circle packing evaluation using rayevolve.eval."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    num_experiment_runs = 1

    # Define a nested function to pass results_dir to the aggregator
    def _aggregator_with_context(
        r: List[Tuple[pd.DataFrame, pd.DataFrame]],
    ) -> Dict[str, Any]:
        return aggregate_train_and_classify(r, results_dir)

    metrics, correct, error_msg = run_rayevolve_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="train_and_classify",
        num_runs=num_experiment_runs,
        validate_fn=validate_prediction,
        aggregate_metrics_fn=_aggregator_with_context,
    )

    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <string_too_long_to_display>")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="data set fittingevaluator using rayevolve.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'train_and_classify_validation')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json, extra.npz)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)
