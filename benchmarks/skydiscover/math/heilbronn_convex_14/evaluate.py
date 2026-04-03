"""Evaluator for the Heilbronn problem for convex regions (14 points)."""

import sys
import uuid
import json
import itertools
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

BENCHMARK = 0.027835571458482138
NUM_POINTS = 14


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


def triangle_area(p1, p2, p3):
    return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2


if __name__ == "__main__":
    module = load_module_from_path("main.py")
    points = module.heilbronn_convex14()

    if not isinstance(points, np.ndarray):
        points = np.array(points)

    error_msg = None
    is_valid = True

    if points.shape != (NUM_POINTS, 2):
        is_valid = False
        error_msg = f"Invalid shape: {points.shape}, expected ({NUM_POINTS}, 2)"

    if is_valid:
        try:
            min_triangle_area = min(
                triangle_area(p1, p2, p3) for p1, p2, p3 in itertools.combinations(points, 3)
            )
            convex_hull_area = ConvexHull(points).volume
            min_area_normalized = min_triangle_area / convex_hull_area
            combined_score = float(min_area_normalized / BENCHMARK)
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
