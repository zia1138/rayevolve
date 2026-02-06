#!/usr/bin/env python3
"""
baseline_eval_env.py

Minimal baseline evaluator.

- Loads graphs via pickle (*.pkl) (tuple nodes preserved)
- Runs an exhaustive BFS baseline through an environment wrapper
- Returns ONLY:
    {
      "total_score": float,
      "valid": bool   # False if ANY instance fails (no valid path)
    }

Dataset layout (default ./data):
- index.json
- graphs/*.pkl
- metas/*.json

Requires:
  pip install networkx typer
"""

import collections
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import networkx as nx
import typer

app = typer.Typer(add_completion=False)


# -------------------------
# Environment
# -------------------------

class SearchEnv:
    def __init__(self, adj: Dict, start, goal):
        self.adj = adj
        self.start = start
        self.goal = goal
        self.nodes_expanded = 0
        self.neighbors_iterated = 0

    def is_goal(self, n) -> bool:
        return n == self.goal

    def expand(self, _n) -> None:
        self.nodes_expanded += 1

    def neighbors(self, n) -> Iterable:
        for nbr in self.adj.get(n, []):
            self.neighbors_iterated += 1
            yield nbr


# -------------------------
# Baseline search (exhaustive BFS)
# -------------------------

def reconstruct_path(came_from: Dict, start, goal) -> List:
    cur = goal
    path = [cur]
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def graph_search(env: SearchEnv) -> Optional[List]:
    start = env.start
    goal = env.goal

    if start not in env.adj or goal not in env.adj:
        return None

    q = collections.deque([start])
    visited = {start}
    came_from = {}
    goal_reached = (start == goal)

    while q:
        cur = q.popleft()
        env.expand(cur)

        if env.is_goal(cur):
            goal_reached = True

        for nbr in env.neighbors(cur):
            if nbr in visited:
                continue
            visited.add(nbr)
            came_from[nbr] = cur
            q.append(nbr)

    if not goal_reached:
        return None
    if start == goal:
        return [start]
    return reconstruct_path(came_from, start, goal)


# -------------------------
# Validation + scoring
# -------------------------

def is_valid_path(adj: Dict, start, goal, path: Optional[List]) -> bool:
    if path is None or not path:
        return False
    if path[0] != start or path[-1] != goal:
        return False
    for a, b in zip(path, path[1:]):
        if b not in adj.get(a, []):
            return False
    return True


def score_instance(
    *,
    found: bool,
    nodes_expanded: int,
    neighbors_iterated: int,
    path_length: int,
    fail_penalty: int = -1_000,
    success_reward: int = 1_000,
    w_expanded: float = 5.0,
    w_neighbors: float = 1.0,
    w_pathlen: float = 10.0,
) -> float:
    if not found:
        return float(fail_penalty)
    return float(
        success_reward
        - w_expanded * nodes_expanded
        - w_neighbors * neighbors_iterated
        - w_pathlen * path_length
    )


# -------------------------
# Evaluation
# -------------------------

def evaluate_all_graphs(dataset_dir: str | Path = "./data") -> Dict:
    dataset_dir = Path(dataset_dir)
    index = json.loads((dataset_dir / "index.json").read_text("utf-8"))

    total_score = 0.0
    all_valid = True

    for inst in index:
        graph_path = dataset_dir / inst["graph"]
        meta_path = dataset_dir / inst["meta"]

        with graph_path.open("rb") as f:
            G: nx.Graph = pickle.load(f)

        meta = json.loads(meta_path.read_text("utf-8"))
        start = tuple(meta["start"])
        goal = tuple(meta["goal"])

        adj = {n: list(G.neighbors(n)) for n in G.nodes}
        env = SearchEnv(adj=adj, start=start, goal=goal)

        path = graph_search(env)
        valid = is_valid_path(adj, start, goal, path)

        if not valid:
            all_valid = False

        path_length = (len(path) - 1) if (path and valid) else 0
        total_score += score_instance(
            found=valid,
            nodes_expanded=env.nodes_expanded,
            neighbors_iterated=env.neighbors_iterated,
            path_length=path_length,
        )

    return {"total_score": total_score, "valid": all_valid}


# -------------------------
# CLI
# -------------------------

@app.command()
def run(
    dataset_dir: Path = typer.Argument(Path("./data"), exists=True, file_okay=False),
):
    out = evaluate_all_graphs(dataset_dir)
    typer.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    app()