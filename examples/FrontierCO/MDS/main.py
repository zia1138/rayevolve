"""Initial solution for the Minimum Dominant Set (MDS) problem."""


def solve(instance_id, graph):
    """
    Solve the Minimum Dominant Set problem for a given test case.

    Args:
        instance_id : (str) Unique identifier for this problem instance.
        graph       : (networkx.Graph) The graph to solve.

    Returns:
        A dict with key "mds_nodes" containing a list of node indices
        in the minimum dominant set.
    """
    # Simple greedy: pick all nodes as a trivially valid dominant set
    mds_nodes = list(graph.nodes())
    return {"mds_nodes": mds_nodes}
