"""Evaluator for the Heilbronn triangle problem (11 points in equilateral triangle)."""

import sys
import uuid
import json
import itertools
import importlib.util
from pathlib import Path

import numpy as np

BENCHMARK = 0.036529889880030156
NUM_POINTS = 11
TOL = 1e-6


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


def triangle_area(a, b, c):
    return abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2


def check_inside_triangle(points, tol=1e-6):
    for x, y in points:
        cond1 = y >= -tol
        cond2 = np.sqrt(3) * x <= np.sqrt(3) - y + tol
        cond3 = y <= np.sqrt(3) * x + tol
        if not (cond1 and cond2 and cond3):
            return False, f"Point ({x}, {y}) is outside the equilateral triangle."
    return True, None


if __name__ == "__main__":
    module = load_module_from_path("main.py")
    points = module.heilbronn_triangle11()

    if not isinstance(points, np.ndarray):
        points = np.array(points)

    error_msg = None
    is_valid = True

    if points.shape != (NUM_POINTS, 2):
        is_valid = False
        error_msg = f"Invalid shape: {points.shape}, expected ({NUM_POINTS}, 2)"
    else:
        is_valid, error_msg = check_inside_triangle(points, TOL)

    if is_valid:
        a = np.array([0, 0])
        b = np.array([1, 0])
        c = np.array([0.5, np.sqrt(3) / 2])
        min_triangle_area = min(
            triangle_area(p1, p2, p3) for p1, p2, p3 in itertools.combinations(points, 3)
        )
        min_area_normalized = min_triangle_area / triangle_area(a, b, c)
        combined_score = float(min_area_normalized / BENCHMARK)
    else:
        combined_score = 0.0

    result = {
        "correct": is_valid,
        "error": error_msg,
        "combined_score": combined_score,
    }
    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
