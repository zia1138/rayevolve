"""Evaluator for circle packing on a rectangle of perimeter 4 (n=21)."""

import sys
import uuid
import json
import importlib.util
from pathlib import Path

import numpy as np

BENCHMARK = 2.3658321334167627
NUM_CIRCLES = 21
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


def minimum_circumscribing_rectangle(circles: np.ndarray):
    min_x = np.min(circles[:, 0] - circles[:, 2])
    max_x = np.max(circles[:, 0] + circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    max_y = np.max(circles[:, 1] + circles[:, 2])
    return max_x - min_x, max_y - min_y


def validate_packing(circles: np.ndarray):
    n = len(circles)
    radii = circles[:, 2]

    for i in range(n):
        if radii[i] < 0:
            return False, f"Circle {i} has negative radius {radii[i]}"
        if np.isnan(radii[i]):
            return False, f"Circle {i} has nan radius"

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((circles[i, :2] - circles[j, :2]) ** 2))
            if dist < circles[i, 2] + circles[j, 2] - TOL:
                return False, (
                    f"Circles {i} and {j} overlap: dist={dist}, "
                    f"r1+r2={circles[i, 2] + circles[j, 2]}"
                )

    width, height = minimum_circumscribing_rectangle(circles)
    if width + height > 2 + TOL:
        return False, "Circles not contained in rectangle of perimeter 4."

    return True, None


if __name__ == "__main__":
    module = load_module_from_path("main.py")
    circles = module.circle_packing21()

    if not isinstance(circles, np.ndarray):
        circles = np.array(circles)

    error_msg = None
    is_valid = True

    if circles.shape != (NUM_CIRCLES, 3):
        is_valid = False
        error_msg = f"Invalid shape: {circles.shape}, expected ({NUM_CIRCLES}, 3)"
    else:
        is_valid, error_msg = validate_packing(circles)

    radii_sum = float(np.sum(circles[:, 2])) if is_valid else 0.0
    combined_score = float(radii_sum / BENCHMARK) if is_valid else 0.0

    result = {
        "correct": is_valid,
        "error": error_msg,
        "combined_score": combined_score,
    }
    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
