"""Evaluator for minimizing max/min distance ratio (dim=2, 16 points)."""

import sys
import uuid
import json
import importlib.util
from pathlib import Path

import numpy as np
import scipy.spatial.distance

NUM_POINTS = 16
DIMENSION = 2
BENCHMARK = 1 / 12.889266112


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


if __name__ == "__main__":
    module = load_module_from_path("main.py")
    points = module.min_max_dist_dim2_16()

    if not isinstance(points, np.ndarray):
        points = np.array(points)

    error_msg = None
    is_valid = True

    if points.shape != (NUM_POINTS, DIMENSION):
        is_valid = False
        error_msg = f"Invalid shape: {points.shape}, expected ({NUM_POINTS}, {DIMENSION})"

    if is_valid:
        try:
            pairwise_distances = scipy.spatial.distance.pdist(points)
            min_distance = np.min(pairwise_distances)
            max_distance = np.max(pairwise_distances)
            inv_ratio_squared = (min_distance / max_distance) ** 2 if max_distance > 0 else 0
            combined_score = float(inv_ratio_squared / BENCHMARK)
        except Exception as e:
            is_valid = False
            error_msg = str(e)
            combined_score = 0.0
    else:
        combined_score = 0.0

    result = {
        "correct": is_valid,
        "error": error_msg,
        "combined_score": combined_score,
    }
    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
