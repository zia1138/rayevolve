import sys
import uuid
import json
import pickle
import importlib.util
from pathlib import Path


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


def evaluate_candidate(candidate_module, dataset_dir: Path) -> dict:
    index = json.loads((dataset_dir / "index.json").read_text("utf-8"))

    class EvaluatorSearchEnv(candidate_module.SearchEnv):
        def __init__(self, adj, start, goal):
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
        G = pickle.loads((dataset_dir / inst["graph"]).read_bytes())
        meta = json.loads((dataset_dir / inst["meta"]).read_text("utf-8"))

        start, goal = tuple(meta["start"]), tuple(meta["goal"])
        adj = {n: list(G.neighbors(n)) for n in G.nodes}

        env = EvaluatorSearchEnv(adj, start, goal)
        path = candidate_module.graph_search(env)

        valid = (path is not None and len(path) > 0 and
                 path[0] == start and path[-1] == goal and
                 all(b in adj.get(a, []) for a, b in zip(path, path[1:])))

        if not valid:
            all_valid = False
            score = 0.0
        else:
            max_cost = 10.0 * sum(len(v) for v in adj.values()) + 1.0 * len(adj)
            actual_cost = 10.0 * env.neighbors_iterated + 1.0 * (len(path) - 1)
            score = max(0.0, max_cost - actual_cost)

        total_score += score

    return {"correct": all_valid, "error": None, "combined_score": total_score}


if __name__ == "__main__":
    candidate_module = load_module_from_path("main.py")
    result = evaluate_candidate(candidate_module, Path("./data"))

    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
