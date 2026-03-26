"""Shared utilities for CO-Bench rayevolve examples."""

import sys
import uuid
import json
import importlib.util
from pathlib import Path


def load_module_from_path(file_path: str | Path, unique: bool = True):
    """Dynamically load a Python module from a file path."""
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


def write_results(result: dict, path: str = "results.json"):
    """Write evaluation results to JSON."""
    with open(path, "w") as f:
        json.dump(result, f, indent=4)
