"""Initial solution for the Travelling Salesman Problem: Nearest Neighbor heuristic."""

import math


def _euclidean_dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def solve(instance_id, nodes):
    """
    Problem:
        Given a set of cities with (x, y) coordinates, find the shortest closed tour
        visiting all cities exactly once and returning to the start. The distance metric
        is Euclidean: sqrt((x2-x1)^2 + (y2-y1)^2).

    Args:
        instance_id : (str) Unique identifier for this problem instance, e.g. "tsp500_test_concorde_0".
        nodes       : (list of tuples) Each tuple is (x, y) coordinates.

    Returns:
        A dictionary with key "tour" containing a 0-indexed permutation of node indices
        representing the order in which to visit the cities.
    """
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
