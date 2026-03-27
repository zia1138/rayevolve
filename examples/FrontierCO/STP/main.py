"""Initial solution for the Steiner Tree Problem in Graphs."""


def solve(instance_id, n, m, graph_edges, terminals):
    """
    Solves the Steiner Tree Problem in Graphs.

    Given an undirected graph with weighted edges, the goal is to compute a subgraph (a tree)
    that connects all the terminal vertices with minimum total cost.

    Important: All vertex indices u, v, and all terminals are 1-indexed, i.e., values range from 1 to n.

    Args:
        instance_id  : (str) Unique identifier for this problem instance.
        n            : (int) Number of vertices in the graph.
        m            : (int) Number of edges in the graph.
        graph_edges  : (dict) Mapping of (min(u,v), max(u,v)) -> cost for each edge.
        terminals    : (list) List of terminal vertices (1-indexed).

    Returns:
        A dictionary with the following keys:
            - 'declared_cost': Total cost of the solution (a number).
            - 'num_edges': Number of edges in the solution (an integer).
            - 'edges': A list of tuples, each tuple (u, v) representing an edge in the solution.
    """
    return {
        'declared_cost': 0.0,
        'num_edges': 0,
        'edges': []
    }
