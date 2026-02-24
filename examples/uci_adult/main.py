import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import confusion_matrix
from rayevolve.core.evaluator import load_module_from_path, save_json_results

app = typer.Typer(add_completion=False)

def calculate_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    """Calculates TPR at FPR = 0.05."""
    FPR_TARGET = 0.05
    
    # Threshold from negative-class quantile
    neg_scores = y_score[y_true == 0]
    thr = float(np.quantile(neg_scores, 1.0 - FPR_TARGET))
    
    # Strictly greater-than to respect the targeted FPR
    thr = np.nextafter(thr, np.inf)
    y_pred = (y_score > thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = float(tp / (tp + fn)) if (tp + fn) else 0.0
    actual_fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0

    return {
        "combined_score": tpr,
        "tpr": tpr,
        "fpr": actual_fpr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

def main(
    program_path: str = typer.Option("main.py"),
    results_dir: str = typer.Option("results"),
    data_dir: str = typer.Option("."),
):
    data_path = Path(data_dir)
    
    # 1. Load Data
    X_train = pd.read_csv(data_path / "X_train.csv", index_col=0)
    y_train = pd.read_csv(data_path / "y_train.csv", index_col=0)
    X_val = pd.read_csv(data_path / "X_val.csv", index_col=0)
    y_val_true = pd.read_csv(data_path / "y_val.csv", index_col=0)

    # 2. Load and Run Candidate
    candidate_module = load_module_from_path(program_path)
    
    # Predict probabilities
    y_val_proba_df = candidate_module.preprocess_train_and_predict(X_train, y_train, X_val)
    
    # Ensure correct format
    y_score = y_val_proba_df["y_proba"].values
    y_true = y_val_true.loc[y_val_proba_df.index].iloc[:, 0].values.astype(int)

    # 3. Score
    metrics = calculate_metrics(y_true, y_score)
    
    # Save for RayEvolve
    save_json_results(results_dir, metrics, correct=True)

if __name__ == "__main__":
    typer.run(main)
