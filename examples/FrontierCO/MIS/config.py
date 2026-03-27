from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

import textwrap

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in combinatorial optimization, specifically the Maximum Independent Set (MIS) problem.

    Problem Description:
    The Maximum Independent Set (MIS) problem is a fundamental NP-hard optimization problem in graph theory. Given an undirected graph G = (V, E), where V is a set of vertices and E is a set of edges, the goal is to find the largest subset S of V such that no two vertices in S are adjacent (i.e., connected by an edge).

    Task:
    - Implement the `solve` function that finds a maximum independent set for a given graph.
    - The solution must be a valid independent set: no two vertices in the set may be adjacent.
    - The objective is to maximize the size of the independent set.
    - If the solution is not a valid independent set, it receives no score.

    Input args:
        instance_id : (str) Unique identifier for this problem instance.
        graph       : (networkx.Graph) The graph to solve.

    Returns:
        A dict with key "mis_nodes" containing a list of node indices in the maximum independent set.

    Key insights to explore:
    1. Greedy selection of vertices with lowest degree
    2. Graph coloring or complement graph approaches
    3. Local search or metaheuristics (simulated annealing, genetic algorithms)
    4. Integer linear programming formulations
    5. Branch and bound with pruning strategies

    IMPORTANT: The main entry point is `def solve(instance_id, graph)`.
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
