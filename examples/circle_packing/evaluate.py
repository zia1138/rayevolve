"""Evaluator for circle packing example (n=26)."""

import sys
import uuid
import json
import importlib.util
from pathlib import Path

import numpy as np


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


def validate_packing(centers, radii, reported_sum, atol=1e-6):
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)

    n_expected = 26
    if centers.shape != (n_expected, 2):
        return False, f"Centers shape incorrect. Expected ({n_expected}, 2), got {centers.shape}"
    if radii.shape != (n_expected,):
        return False, f"Radii shape incorrect. Expected ({n_expected},), got {radii.shape}"

    if np.any(radii < 0):
        negative_indices = np.where(radii < 0)[0]
        return False, f"Negative radii found for circles at indices: {negative_indices}"

    if not np.isclose(np.sum(radii), reported_sum, atol=atol):
        return False, (
            f"Sum of radii ({np.sum(radii):.6f}) does not match "
            f"reported ({reported_sum:.6f})"
        )

    for i in range(n_expected):
        x, y = centers[i]
        r = radii[i]
        if x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol:
            return False, f"Circle {i} (x={x:.4f}, y={y:.4f}, r={r:.4f}) is outside unit square."

    for i in range(n_expected):
        for j in range(i + 1, n_expected):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:
                return False, (
                    f"Circles {i} & {j} overlap. Dist: {dist:.4f}, "
                    f"Sum Radii: {(radii[i] + radii[j]):.4f}"
                )

    return True, None


if __name__ == "__main__":
    module = load_module_from_path("main.py")
    centers, radii, reported_sum = module.run_packing()
    is_valid, error_msg = validate_packing(centers, radii, reported_sum)

    if not is_valid:
        print(error_msg)

    result = {"correct": is_valid, "error": error_msg, "combined_score": float(reported_sum)}
    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
