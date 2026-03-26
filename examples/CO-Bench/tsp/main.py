"""Seed heuristic for TSP: Nearest Neighbor."""

import math


def _euclidean_dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def solve(**kwargs):
    """
    Solve a TSP instance.

    Input kwargs:
      - nodes (list): List of (x, y) coordinates

    Returns:
      dict with:
        - 'tour': 0-indexed permutation of node indices
    """
    nodes = kwargs["nodes"]
    n = len(nodes)

    if n == 0:
        return {"tour": []}

    # Nearest neighbor starting from node 0
    visited = [False] * n
    tour = [0]
    visited[0] = True

    for _ in range(n - 1):
        current = tour[-1]
        best_dist = float("inf")
        best_next = -1

        for j in range(n):
            if not visited[j]:
                d = _euclidean_dist(nodes[current], nodes[j])
                if d < best_dist:
                    best_dist = d
                    best_next = j

        tour.append(best_next)
        visited[best_next] = True

    return {"tour": tour}
