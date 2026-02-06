#!/usr/bin/env python3
"""
Generate grid-based graph benchmarks for pruning-sensitive graph search.

Families:
1) plain_grid: random obstacles
2) detour_gate: vertical wall with one off-center gate

Outputs (under --out DIR):
- graphs/<name>.pkl          (pickle'd NetworkX graph with tuple nodes)
- metas/<name>.json          (start/goal + generation params)
- index.json                 (list of instances)

Optional:
- viz/<name>.png             (grid visualization; requires matplotlib)
- graphml/<name>.graphml     (for Gephi / yEd)

Install:
  pip install networkx typer
Optional (for PNGs):
  pip install matplotlib
"""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import typer

app = typer.Typer(add_completion=False)

Node = Tuple[int, int]


# -------------------------
# Metadata
# -------------------------

@dataclass
class InstanceMeta:
    name: str
    family: str
    width: int
    height: int
    obstacle_p: float
    seed: int
    start: Node
    goal: Node
    wall_x: Optional[int] = None
    gate_y: Optional[int] = None


# -------------------------
# Helpers
# -------------------------

def in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def neighbors4(x: int, y: int) -> List[Node]:
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def choose_start_goal(w: int, h: int) -> Tuple[Node, Node]:
    return (2, h // 2), (w - 3, h // 2)


def build_grid_graph(w: int, h: int, blocked: Set[Node]) -> nx.Graph:
    G = nx.Graph()
    for x in range(w):
        for y in range(h):
            if (x, y) in blocked:
                continue
            G.add_node((x, y))
    for x in range(w):
        for y in range(h):
            if (x, y) in blocked:
                continue
            for nx_, ny_ in neighbors4(x, y):
                if in_bounds(nx_, ny_, w, h) and (nx_, ny_) not in blocked:
                    G.add_edge((x, y), (nx_, ny_))
    return G


def ensure_path_exists(
    rng: random.Random,
    w: int,
    h: int,
    obstacle_p: float,
    start: Node,
    goal: Node,
    forced_blocked: Optional[Set[Node]] = None,
    max_tries: int = 300,
) -> Set[Node]:
    forced_blocked = forced_blocked or set()
    keep = {start, goal}

    for _ in range(max_tries):
        blocked = set(forced_blocked)
        for x in range(w):
            for y in range(h):
                n = (x, y)
                if n in blocked or n in keep:
                    continue
                if rng.random() < obstacle_p:
                    blocked.add(n)

        G = build_grid_graph(w, h, blocked)
        if start in G and goal in G and nx.has_path(G, start, goal):
            return blocked

    raise RuntimeError("Failed to generate a connected instance.")


# -------------------------
# Graph families
# -------------------------

def make_plain_grid(rng, name, w, h, obstacle_p, seed):
    start, goal = choose_start_goal(w, h)
    blocked = ensure_path_exists(rng, w, h, obstacle_p, start, goal)
    G = build_grid_graph(w, h, blocked)

    meta = InstanceMeta(
        name=name,
        family="plain_grid",
        width=w,
        height=h,
        obstacle_p=obstacle_p,
        seed=seed,
        start=start,
        goal=goal,
    )
    G.graph["blocked"] = list(map(list, blocked))
    return G, meta


def make_detour_gate(rng, name, w, h, obstacle_p, seed, gate_band=10):
    start, goal = choose_start_goal(w, h)
    wall_x = w // 2

    if rng.random() < 0.5:
        gate_y = rng.randint(0, min(gate_band - 1, h - 1))
    else:
        gate_y = rng.randint(max(0, h - gate_band), h - 1)

    forced_blocked = {(wall_x, y) for y in range(h) if y != gate_y}

    blocked = ensure_path_exists(rng, w, h, obstacle_p, start, goal, forced_blocked)
    G = build_grid_graph(w, h, blocked)

    meta = InstanceMeta(
        name=name,
        family="detour_gate",
        width=w,
        height=h,
        obstacle_p=obstacle_p,
        seed=seed,
        start=start,
        goal=goal,
        wall_x=wall_x,
        gate_y=gate_y,
    )
    G.graph["blocked"] = list(map(list, blocked))
    return G, meta


# -------------------------
# Visualization / GraphML
# -------------------------

def save_png_grid(out_png: Path, meta: InstanceMeta, blocked_cells: List[List[int]]) -> None:
    """
    Save a simple grid visualization:
    - blocked cells in black
    - free cells in white
    - start (circle), goal (star)
    - for detour_gate: dashed wall line and square gate marker
    """
    import matplotlib.pyplot as plt  # optional dependency

    w, h = meta.width, meta.height
    sx, sy = meta.start
    gx, gy = meta.goal

    grid = [[0] * w for _ in range(h)]
    for x, y in blocked_cells:
        if 0 <= x < w and 0 <= y < h:
            grid[y][x] = 1

    fig, ax = plt.subplots()
    ax.imshow(grid, origin="lower")

    ax.scatter([sx], [sy], marker="o", s=60, label="start")
    ax.scatter([gx], [gy], marker="*", s=120, label="goal")

    if meta.family == "detour_gate" and meta.wall_x is not None and meta.gate_y is not None:
        ax.axvline(meta.wall_x, linestyle="--")
        ax.scatter([meta.wall_x], [meta.gate_y], marker="s", s=80, label="gate")

    ax.set_title(f"{meta.name} | {meta.family}")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_graphml(out_graphml: Path, G: nx.Graph, meta: InstanceMeta) -> None:
    """
    Save GraphML for external visualization tools (Gephi/yEd).
    Convert tuple nodes (x,y) to string ids "x,y" and attach x,y attributes.
    """
    H = nx.Graph()
    for (x, y) in G.nodes:
        nid = f"{x},{y}"
        H.add_node(nid, x=int(x), y=int(y))
    for (a, b) in G.edges:
        ax, ay = a
        bx, by = b
        H.add_edge(f"{ax},{ay}", f"{bx},{by}")

    # graph-level metadata (scalar only)
    H.graph["name"] = meta.name
    H.graph["family"] = meta.family
    H.graph["width"] = meta.width
    H.graph["height"] = meta.height
    H.graph["obstacle_p"] = meta.obstacle_p
    H.graph["seed"] = meta.seed
    H.graph["start"] = f"{meta.start[0]},{meta.start[1]}"
    H.graph["goal"] = f"{meta.goal[0]},{meta.goal[1]}"
    if meta.wall_x is not None:
        H.graph["wall_x"] = meta.wall_x
    if meta.gate_y is not None:
        H.graph["gate_y"] = meta.gate_y

    out_graphml.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(H, out_graphml)


# -------------------------
# Saving
# -------------------------

def save_instance(
    out_dir: Path,
    G: nx.Graph,
    meta: InstanceMeta,
    *,
    write_png: bool,
    write_graphml_flag: bool,
) -> Dict:
    graphs_dir = out_dir / "graphs"
    metas_dir = out_dir / "metas"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    metas_dir.mkdir(parents=True, exist_ok=True)

    # graph pickle
    gpath = graphs_dir / f"{meta.name}.pkl"
    with gpath.open("wb") as f:
        pickle.dump(G, f)

    # meta json
    mpath = metas_dir / f"{meta.name}.json"
    meta_dict = asdict(meta)
    meta_dict["start"] = list(meta.start)
    meta_dict["goal"] = list(meta.goal)
    mpath.write_text(json.dumps(meta_dict, indent=2), "utf-8")

    # optional viz
    if write_png:
        blocked_cells = G.graph.get("blocked", [])
        save_png_grid(out_dir / "viz" / f"{meta.name}.png", meta, blocked_cells)

    # optional graphml
    if write_graphml_flag:
        save_graphml(out_dir / "graphml" / f"{meta.name}.graphml", G, meta)

    return {
        "name": meta.name,
        "family": meta.family,
        "graph": f"graphs/{meta.name}.pkl",
        "meta": f"metas/{meta.name}.json",
        "seed": meta.seed,
        "viz": f"viz/{meta.name}.png" if write_png else None,
        "graphml": f"graphml/{meta.name}.graphml" if write_graphml_flag else None,
    }


# -------------------------
# CLI
# -------------------------

@app.command()
def generate(
    out: Path = typer.Option(...),
    width: int = typer.Option(20),
    height: int = typer.Option(20),
    obstacle_p: float = typer.Option(0.2),
    seed: int = typer.Option(0),
    n_plain: int = typer.Option(3),
    n_gate: int = typer.Option(3),
    gate_band: int = typer.Option(10, help="Top/bottom band height for placing the gate"),
    write_png: bool = typer.Option(True, help="Write PNGs to out/viz (requires matplotlib)"),
    write_graphml_flag: bool = typer.Option(False, help="Write GraphML to out/graphml (Gephi/yEd)"),
):
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)

    index: List[Dict] = []

    for i in range(n_plain):
        s = rng.randint(0, 10**9)
        G, meta = make_plain_grid(random.Random(s), f"plain_{i:04d}", width, height, obstacle_p, s)
        index.append(save_instance(out, G, meta, write_png=write_png, write_graphml_flag=write_graphml_flag))

    for i in range(n_gate):
        s = rng.randint(0, 10**9)
        G, meta = make_detour_gate(random.Random(s), f"gate_{i:04d}", width, height, obstacle_p, s, gate_band=gate_band)
        index.append(save_instance(out, G, meta, write_png=write_png, write_graphml_flag=write_graphml_flag))

    (out / "index.json").write_text(json.dumps(index, indent=2), "utf-8")
    typer.echo(f"Wrote {len(index)} graphs to {out}")


if __name__ == "__main__":
    app()