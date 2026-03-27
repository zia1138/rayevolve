from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

import textwrap

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in combinatorial optimization, specifically the Capacitated P-Median Problem.

    Problem Description:
    The Capacitated P-Median Problem is a facility location optimization problem where the objective is to select exactly p customers as medians (facility locations) and assign each customer to one of these medians to minimize the total cost, defined as the sum of the Euclidean distances between customers and their assigned medians. Each customer has a capacity Q_i and demand q_i. If a customer is selected as the median, the total demand of the customers assigned to it cannot exceed its capacity Q_i. A feasible solution must respect this capacity constraint for all medians. Note that each customer should be assigned to exactly one median, including the customers which are selected as the median.

    Task:
    - Implement the `solve` function that selects p medians and assigns all customers to medians.
    - Each median has a capacity constraint on total assigned demand.
    - The objective is to minimize the total Euclidean distance cost.
    - If any constraint is violated, the solution receives no score.

    Input args:
        instance_id : (str) Unique identifier for this problem instance.
        n           : (int) Number of customers/points.
        p           : (int) Number of medians to choose.
        customers   : (list of tuples) Each tuple is (customer_id, x, y, capacity, demand).
                      Note: capacity is only relevant if the point is selected as a median.

    Returns:
        A dict with keys:
            "objective"   : (numeric) The total cost (objective value).
            "medians"     : (list of int) Exactly p customer IDs chosen as medians.
            "assignments" : (list of int) A list of n integers, where the i-th integer is the
                            customer ID (from the chosen medians) assigned to customer i.

    Key insights to explore:
    1. Greedy median selection based on centrality
    2. K-means style iterative assignment and re-selection
    3. Local search swapping medians
    4. Lagrangian relaxation of capacity constraints
    5. Tabu search or simulated annealing metaheuristics
    6. Constructive heuristics with capacity-aware assignments

    IMPORTANT: The main entry point is `def solve(instance_id, n, p, customers)`.
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
