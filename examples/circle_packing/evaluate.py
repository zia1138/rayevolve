"""
Evaluator for circle packing example (n=26) with improved timeout handling
"""

import typer
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from rayevolve.core.evaluator import load_module_from_path, save_json_results

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
        "initial.py",
        help="Path to program to evaluate (must contain 'run_packing')",
    ),
    results_dir: str = typer.Option(
        "results",
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
