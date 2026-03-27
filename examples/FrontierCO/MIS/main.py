"""Initial solution for the Maximum Independent Set (MIS) problem."""


def solve(instance_id, graph):
    """
    Solve the Maximum Independent Set problem for a given test case.

    Args:
        instance_id : (str) Unique identifier for this problem instance.
        graph       : (networkx.Graph) The graph to solve.

    Returns:
        A dict with key "mis_nodes" containing a list of node indices
        in the maximum independent set.
    """
    # Simple greedy: pick nodes with lowest degree, removing neighbors
    remaining = set(graph.nodes())
    mis_nodes = []
    while remaining:
        # Pick the node with the smallest degree among remaining nodes
        node = min(remaining, key=lambda n: len(set(graph.neighbors(n)) & remaining))
        mis_nodes.append(node)
        # Remove the node and its neighbors from remaining
        remaining.discard(node)
        remaining -= set(graph.neighbors(node))
    return {"mis_nodes": mis_nodes}
