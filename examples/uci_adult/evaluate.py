import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import confusion_matrix

app = typer.Typer(add_completion=False)

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

def save_json_results(
    results_dir: str | Path,
    metrics: Dict[str, Any],
    correct: bool,
    error: Optional[str] = None,
) -> None:
    """Saves metrics and correctness status to JSON files."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    correct_payload = {"correct": correct, "error": error}
    correct_file = results_path / "correct.json"
    with open(correct_file, "w") as f:
        json.dump(correct_payload, f, indent=4)
    #print(f"Correctness and error status saved to {correct_file}")

    metrics_file = results_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    #print(f"Metrics saved to {metrics_file}")    

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
