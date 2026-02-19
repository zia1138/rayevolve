from pathlib import Path
import importlib.util
import sys
import uuid

import json
from typing import Any, Dict, Optional

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