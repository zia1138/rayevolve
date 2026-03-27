from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

import textwrap

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in combinatorial optimization, specifically the Capacitated Vehicle Routing Problem (CVRP).

    Problem Description:
    The Capacitated Vehicle Routing Problem (CVRP) is a classic optimization problem that extends the Traveling Salesman Problem. In the CVRP, a fleet of vehicles with limited capacity must service a set of customers with specific demands, starting and ending at a central depot. Each customer must be visited exactly once by exactly one vehicle, and the total demand of customers on a single vehicle's route cannot exceed the vehicle's capacity. The objective is to minimize the total travel distance while satisfying all customer demands and vehicle capacity constraints.

    Task:
    - Implement the `solve` function that finds vehicle routes to service all customers.
    - Each customer must be visited exactly once by exactly one vehicle.
    - Each route must start and end at the depot.
    - The total demand on each route must not exceed the vehicle capacity.
    - The objective is to minimize total travel distance.
    - If any constraint is violated, the solution receives no score.

    Input args:
        instance_id : (str) Unique identifier for this problem instance.
        nodes       : (list) List of (x, y) coordinates representing locations (depot and customers).
        demands     : (list) List of customer demands, where demands[i] is the demand for node i.
        capacity    : (int) Vehicle capacity.
        depot_idx   : (int) Index of the depot in the nodes list (typically 0).

    Returns:
        A dict with key "routes" containing a list of routes, where each route is a list of
        node indices starting and ending at the depot.
        Format: {"routes": [[0, 3, 1, 0], [0, 2, 5, 0], ...]}

    Key insights to explore:
    1. Nearest-neighbor or savings-based construction heuristics
    2. Route splitting and merging strategies
    3. Local search moves (2-opt, or-opt, relocate, exchange)
    4. Metaheuristics (simulated annealing, genetic algorithms, tabu search)
    5. Cluster-first, route-second approaches
    6. Capacity-aware insertion heuristics

    IMPORTANT: The main entry point is `def solve(instance_id, nodes, demands, capacity, depot_idx)`.
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
