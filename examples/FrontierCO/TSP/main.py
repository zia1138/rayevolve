"""Initial solution for the Traveling Salesman Problem."""


def solve(instance_id, nodes):
    """
    Solve a TSP instance.

    Given a set of cities with known coordinates, find the shortest possible tour
    that visits each city exactly once and returns to the starting city.

    Args:
        instance_id : (str) Unique identifier for this problem instance.
        nodes       : (list) List of (x, y) coordinates representing cities.
                      Format: [(x1, y1), (x2, y2), ..., (xn, yn)]

    Returns:
        A dictionary with key:
            - 'tour': List of node indices representing the solution path.
                      Format: [0, 3, 1, ...] where numbers are indices into the nodes list.
    """
    return {
        'tour': []
    }
