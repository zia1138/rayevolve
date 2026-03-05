"""
Evaluator for circle packing example (n=26) with improved timeout handling
"""

import typer
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import sys
import uuid
import importlib.util
import json

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

def adapted_validate_packing(
    run_output: Tuple[np.ndarray, np.ndarray, float],
    atol=1e-6,
) -> Tuple[bool, Optional[str]]:
    """
    Validates circle packing results based on the output of 'run_packing'.

    Args:
        run_output: Tuple (centers, radii, reported_sum) from run_packing.

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    centers, radii, reported_sum = run_output
    msg = "The circles are placed correctly. There are no overlaps or any circles outside the unit square."
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)

    n_expected = 26
    if centers.shape != (n_expected, 2):
        msg = (
            f"Centers shape incorrect. Expected ({n_expected}, 2), got {centers.shape}"
        )
        return False, msg
    if radii.shape != (n_expected,):
        msg = f"Radii shape incorrect. Expected ({n_expected},), got {radii.shape}"
        return False, msg

    if np.any(radii < 0):
        negative_indices = np.where(radii < 0)[0]
        msg = f"Negative radii found for circles at indices: {negative_indices}"
        return False, msg

    if not np.isclose(np.sum(radii), reported_sum, atol=atol):
        msg = (
            f"Sum of radii ({np.sum(radii):.6f}) does not match "
            f"reported ({reported_sum:.6f})"
        )
        return False, msg

    for i in range(n_expected):
        x, y = centers[i]
        r = radii[i]
        is_outside = (
            x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol
        )
        if is_outside:
            msg = (
                f"Circle {i} (x={x:.4f}, y={y:.4f}, r={r:.4f}) is outside unit square."
            )
            return False, msg

    for i in range(n_expected):
        for j in range(i + 1, n_expected):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:
                msg = (
                    f"Circles {i} & {j} overlap. Dist: {dist:.4f}, "
                    f"Sum Radii: {(radii[i] + radii[j]):.4f}"
                )
                return False, msg
    return True, msg


def main(
    program_path: str = typer.Option(
        "main.py",
        help="Path to program to evaluate (must contain 'run_packing')",
    ),
    results_dir: str = typer.Option(
        "./",
        help="Dir to save results (metrics.json and correct.json)",
    ),
):
    """Runs the circle packing evaluation using rayevolve.eval."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    module = load_module_from_path(program_path)

    centers, radii, reported_sum = module.run_packing()
    is_valid, error_msg = adapted_validate_packing((centers, radii, reported_sum))
    
    if not is_valid:
        print(error_msg)

    # Save metrics.json and correct.json.
    save_json_results(results_dir, {"combined_score": float(reported_sum)}, is_valid, error_msg)

app = typer.Typer(add_completion=False)
app.command()(main)

if __name__ == "__main__":
    app()
