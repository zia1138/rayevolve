import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Any

import networkx as nx
import typer

app = typer.Typer(add_completion=False)

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

def evaluate_candidate(candidate_module, dataset_dir: Path) -> Dict[str, Any]:
    index = json.loads((dataset_dir / "index.json").read_text("utf-8"))

    # 1. Inherit from the candidate's SearchEnv to ensure type compatibility
    # This allows 'isinstance(env, SearchEnv)' to work inside the candidate.
    class EvaluatorSearchEnv(candidate_module.SearchEnv):
        def __init__(self, adj: Dict, start, goal):
            self._adj = adj
            self._start = start
            self._goal = goal
            self.neighbors_iterated = 0

        @property
        def start(self): return self._start

        @property
        def goal(self): return self._goal

        def is_goal(self, n): return n == self._goal

        def neighbors(self, n):
            for nbr in self._adj.get(n, []):
                self.neighbors_iterated += 1
                yield nbr

    total_score = 0.0
    all_valid = True

    for inst in index:
        # Load Graph and Meta
        G = pickle.loads((dataset_dir / inst["graph"]).read_bytes())
        meta = json.loads((dataset_dir / inst["meta"]).read_text("utf-8"))
        
        start, goal = tuple(meta["start"]), tuple(meta["goal"])
        adj = {n: list(G.neighbors(n)) for n in G.nodes}

        # 2. Run the candidate algorithm
        env = EvaluatorSearchEnv(adj, start, goal)
        path = candidate_module.graph_search(env)

        # 3. Validation
        valid = (path is not None and len(path) > 0 and 
                 path[0] == start and path[-1] == goal and 
                 all(b in adj.get(a, []) for a, b in zip(path, path[1:])))
        
        if not valid:
            all_valid = False
            score = 0.0
        else:
            # Score = (Max Possible Cost) - (Actual Cost)
            # Weights: 10.0 per neighbor query, 1.0 per path edge
            max_cost = 10.0 * sum(len(v) for v in adj.values()) + 1.0 * len(adj)
            actual_cost = 10.0 * env.neighbors_iterated + 1.0 * (len(path) - 1)
            score = max(0.0, max_cost - actual_cost)

        total_score += score

    return {"combined_score": total_score, "valid": all_valid}

def main(
    program_path: str = typer.Option("main.py"),
    results_dir: str = typer.Option("results"),
    dataset_dir: str = typer.Option("./data"),
):
    # Load candidate as a module
    candidate_module = load_module_from_path(program_path)
    
    # Run evaluation
    metrics = evaluate_candidate(candidate_module, Path(dataset_dir))
    
    # Save results for RayEvolve
    is_valid = metrics.pop("valid")
    save_json_results(results_dir, metrics, is_valid)

if __name__ == "__main__":
    typer.run(main)
