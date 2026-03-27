from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

import textwrap

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in combinatorial optimization, specifically the Capacitated Facility Location Problem.

    Problem Description:
    The Capacitated Facility Location Problem aims to determine which facilities to open and how to allocate portions of customer demands among these facilities in order to minimize total costs. Given a set of potential facility locations, each with a fixed opening cost and capacity limit, and a set of customers with individual demands and associated assignment costs to each facility, the objective is to decide which facilities to open and how to distribute each customer's demand among these open facilities. The allocation must satisfy the constraint that the sum of portions assigned to each customer equals their total demand, and that the total demand allocated to any facility does not exceed its capacity. The optimization seeks to minimize the sum of fixed facility opening costs and the total assignment costs. However, if any solution violates these constraints (i.e., a customer's demand is not fully satisfied or a warehouse's capacity is exceeded), then an infinitely large cost is given.

    Task:
    - Implement the `solve` function that determines which facilities to open and how to assign customer demands.
    - Each facility has a fixed opening cost and a capacity limit.
    - Each customer has a demand that must be fully satisfied.
    - The objective is to minimize the total cost (fixed + transportation) while satisfying all constraints.
    - If any constraint is violated, the solution receives no score.

    Input args:
        instance_id  : (str) Unique identifier for this problem instance.
        n            : (int) Number of facilities.
        m            : (int) Number of customers.
        capacities   : (list) A list of capacities for each facility.
        fixed_cost   : (list) A list of fixed costs for each facility.
        demands      : (list) A list of demands for each customer.
        trans_costs  : (list of list) A 2D list of transportation costs, where trans_costs[i][j] represents
                       the cost of allocating the entire demand of customer j to facility i.

    Returns:
        A dict with keys:
            "total_cost"       : (float) The computed objective value.
            "facilities_open"  : (list of int) A list of n integers (0 or 1) indicating open/closed.
            "assignments"      : (list of list of float) A 2D list (m x n) of demand allocations.

    Key insights to explore:
    1. Greedy facility opening based on cost-to-capacity ratio
    2. Linear programming relaxation for fractional assignments
    3. Local search swapping open/closed facilities
    4. Lagrangian relaxation approaches
    5. Column generation techniques
    6. Rounding heuristics from LP relaxation solutions

    IMPORTANT: The main entry point is `def solve(instance_id, n, m, capacities, fixed_cost, demands, trans_costs)`.
""")


def list_profiles() -> list[str]:
    """List available configuration profiles to display on CLI."""
    return ["default"]


def build_strategy_model() -> ModelSpec:
    return ModelSpec(
        description="GEMINI 3 Flash Preview",
        model=GoogleModel("gemini-3-flash-preview"),
        settings=GoogleModelSettings(),
    )


def build_evo_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            description="GEMINI 3 Flash Preview",
            model=GoogleModel("gemini-3-flash-preview"),
            settings=GoogleModelSettings(google_thinking_config={"thinking_budget": 8192})
        )
    ]


def get_config(profile: str = "default") -> RayEvolveConfig:
    """Get configuration for the given profile."""
    if profile == "default":
        return RayEvolveConfig(
            evo=EvolutionConfig(task_sys_msg=SYSTEM_MSG,
                                build_strategy_model=build_strategy_model,
                                build_evo_models=build_evo_models,
                                num_agent_workers=1),
            backend=BackendConfig(),
        )
    raise ValueError(f"Unknown profile: {profile}")
