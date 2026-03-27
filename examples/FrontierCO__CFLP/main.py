"""Initial solution for the Capacitated Facility Location Problem."""


def solve(instance_id, n, m, capacities, fixed_cost, demands, trans_costs):
    """
    Problem:
        Given n potential facility locations and m customers, decide which facilities
        to open and how to assign customer demands to minimize total cost (fixed
        opening costs + transportation costs), subject to:
        - Each customer's demand must be fully satisfied.
        - No allocation to closed facilities.
        - Total allocation to each facility must not exceed its capacity.

    Args:
        instance_id  : (str) Unique identifier for this problem instance.
        n            : (int) Number of potential facilities.
        m            : (int) Number of customers.
        capacities   : (list of float) Capacity of each facility, length n.
        fixed_cost   : (list of float) Fixed cost for opening each facility, length n.
        demands      : (list of float) Demand of each customer, length m.
        trans_costs  : (list of lists) trans_costs[i][j] is the cost of serving
                       customer j's ENTIRE demand from facility i. For partial
                       allocation of a units, cost = (a / demands[j]) * trans_costs[i][j].
                       Shape: n x m.

    Returns:
        A dict with keys:
            "total_cost"       : (float) The computed total cost.
            "facilities_open"  : (list of int) Binary list of length n (1=open, 0=closed).
            "assignments"      : (list of lists) assignments[j][i] is the amount of
                                 customer j's demand served by facility i. Shape: m x n.
    """
    # Trivial greedy: open all facilities, assign each customer to cheapest facility
    facilities_open = [1] * n
    remaining_cap = list(capacities)
    assignments = [[0.0] * n for _ in range(m)]

    for j in range(m):
        demand_left = demands[j]
        # Sort facilities by transport cost for this customer
        order = sorted(range(n), key=lambda i: trans_costs[i][j])
        for i in order:
            if demand_left <= 0:
                break
            alloc = min(demand_left, remaining_cap[i])
            assignments[j][i] = alloc
            remaining_cap[i] -= alloc
            demand_left -= alloc

    # Compute total cost
    total_cost = sum(fc for fc, yo in zip(fixed_cost, facilities_open) if yo == 1)
    for j in range(m):
        if demands[j] > 0:
            for i in range(n):
                total_cost += (assignments[j][i] / demands[j]) * trans_costs[i][j]

    return {
        "total_cost": total_cost,
        "facilities_open": facilities_open,
        "assignments": assignments,
    }
