from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

import textwrap

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in combinatorial optimization, specifically the Minimum Dominant Set (MDS) problem.

    Problem Description:
    The Minimum Dominant Set (MDS) problem is a fundamental NP-hard optimization problem in graph theory. Given an undirected graph G = (V, E), where V is a set of vertices and E is a set of edges, the goal is to find the smallest subset D of V such that every vertex in V is either in D or adjacent to at least one vertex in D.

    Task:
    - Implement the `solve` function that finds a minimum dominant set for a given graph.
    - The solution must be a valid dominant set: every vertex must be either in the set or adjacent to at least one vertex in the set.
    - The objective is to minimize the size of the dominant set.
    - If the solution is not a valid dominant set, it receives no score.

    Input args:
        instance_id : (str) Unique identifier for this problem instance.
        graph       : (networkx.Graph) The graph to solve.

    Returns:
        A dict with key "mds_nodes" containing a list of node indices in the minimum dominant set.

    Key insights to explore:
    1. Greedy selection of vertices with highest degree
    2. Graph coloring or clustering-based approaches
    3. Local search or metaheuristics (simulated annealing, genetic algorithms)
    4. Integer linear programming formulations
    5. Iterative vertex elimination strategies

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
