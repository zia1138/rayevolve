"""Initial solution for the Capacitated Vehicle Routing Problem (CVRP)."""


def solve(instance_id, nodes, demands, capacity, depot_idx):
    """
    Solve a CVRP instance.

    Args:
        instance_id : (str) Unique identifier for this problem instance.
        nodes       : (list) List of (x, y) coordinates representing locations (depot and customers).
                      Format: [(x1, y1), (x2, y2), ..., (xn, yn)]
        demands     : (list) List of customer demands, where demands[i] is the demand for node i.
                      Format: [d0, d1, d2, ..., dn]
        capacity    : (int) Vehicle capacity.
        depot_idx   : (int) Index of the depot in the nodes list (typically 0).

    Returns:
        A dict with key "routes" containing a list of routes, where each route is a list of
        node indices starting and ending at the depot.
        Format: {"routes": [[0, 3, 1, 0], [0, 2, 5, 0], ...]}
    """
    # Simple greedy: one customer per route
    routes = []
    for i in range(len(nodes)):
        if i == depot_idx:
            continue
        routes.append([depot_idx, i, depot_idx])
    return {"routes": routes}
