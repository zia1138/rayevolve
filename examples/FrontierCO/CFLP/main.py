"""Initial solution for the Capacitated Facility Location Problem."""


def solve(instance_id, n, m, capacities, fixed_cost, demands, trans_costs):
    """
    Solves the Capacitated Facility Location Problem.

    Args:
        instance_id  : (str) Unique identifier for this problem instance.
        n            : (int) Number of facilities.
        m            : (int) Number of customers.
        capacities   : (list) A list of capacities for each facility.
        fixed_cost   : (list) A list of fixed costs for each facility.
        demands      : (list) A list of demands for each customer.
        trans_costs  : (list of list) A 2D list of transportation costs, where trans_costs[i][j] represents
                       the cost of allocating the entire demand of customer j to facility i.

    Returns:
        A dictionary with the following keys:
            'total_cost': (float) The computed objective value (cost) if the solution is feasible.
            'facilities_open': (list of int) A list of n integers (0 or 1) indicating whether each facility is open.
            'assignments': (list of list of float) A 2D list (m x n) where each entry represents the amount
                           of customer i's demand supplied by facility j.
    """
    # Simple placeholder: open all facilities, assign each customer to cheapest facility
    facilities_open = [1] * n
    assignments = [[0.0] * n for _ in range(m)]
    for j in range(m):
        # Find cheapest facility for this customer
        best_i = min(range(n), key=lambda i: trans_costs[i][j])
        assignments[j][best_i] = demands[j]

    total_cost = sum(fixed_cost[i] for i in range(n) if facilities_open[i] == 1)
    for j in range(m):
        for i in range(n):
            if assignments[j][i] > 0 and demands[j] > 0:
                fraction = assignments[j][i] / demands[j]
                total_cost += fraction * trans_costs[i][j]

    return {
        "total_cost": total_cost,
        "facilities_open": facilities_open,
        "assignments": assignments,
    }
