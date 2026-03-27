"""Initial solution for the Capacitated P-Median Problem."""


def solve(instance_id, n, p, customers):
    """
    Solve the Capacitated P-Median Problem.

    Args:
        instance_id : (str) Unique identifier for this problem instance.
        n           : (int) Number of customers/points.
        p           : (int) Number of medians to choose.
        customers   : (list of tuples) Each tuple is (customer_id, x, y, capacity, demand).
                      Note: capacity is only relevant if the point is selected as a median.

    The goal is to select p medians (from the customers) and assign every customer to one
    of these medians so that the total cost is minimized. The cost for a customer is the
    Euclidean distance to its assigned median, and the total demand assigned to each median
    must not exceed its capacity.

    Returns:
        A dictionary with the following keys:
            'objective': (numeric) the total cost (objective value).
            'medians': (list of int) exactly p customer IDs chosen as medians.
            'assignments': (list of int) a list of n integers, where the i-th integer is the
                           customer ID (from the chosen medians) assigned to customer i.
    """
    # Simple placeholder: pick first p customers as medians, assign everyone to first median
    medians = [customers[i][0] for i in range(p)]
    assignments = [medians[0]] * n

    return {
        "objective": 0,
        "medians": medians,
        "assignments": assignments,
    }
