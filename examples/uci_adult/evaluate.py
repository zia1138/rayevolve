import sys
import uuid
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


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


def calculate_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Calculates TPR at FPR = 0.05."""
    FPR_TARGET = 0.05

    neg_scores = y_score[y_true == 0]
    thr = float(np.quantile(neg_scores, 1.0 - FPR_TARGET))

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
        "fn": int(fn),
    }


if __name__ == "__main__":
    X_train = pd.read_csv("X_train.csv", index_col=0)
    y_train = pd.read_csv("y_train.csv", index_col=0)
    X_val = pd.read_csv("X_val.csv", index_col=0)
    y_val_true = pd.read_csv("y_val.csv", index_col=0)

    candidate_module = load_module_from_path("main.py")
    y_val_proba_df = candidate_module.preprocess_train_and_predict(X_train, y_train, X_val)

    y_score = y_val_proba_df["y_proba"].values
    y_true = y_val_true.loc[y_val_proba_df.index].iloc[:, 0].values.astype(int)

    metrics = calculate_metrics(y_true, y_score)
    result = {"correct": True, "error": None, **metrics}

    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
