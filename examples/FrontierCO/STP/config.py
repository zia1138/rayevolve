from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

import textwrap

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in combinatorial optimization, specifically the Steiner Tree Problem in Graphs.

    Problem Description:
    The Steiner Tree Problem in Graphs requires finding a minimum-cost subgraph that spans a specified set of terminal vertices. Formally, given an undirected graph defined by its vertex count (n), edge count (m), a mapping of edge pairs to weights (graph_edges), and a list of terminal vertices (terminals), the objective is to select a subset of edges that connects all the terminal vertices while minimizing the sum of the edge weights. The solution must consist solely of edges present in the input graph and form a tree that satisfies connectivity among the terminals, with the total cost of the chosen edges equaling the declared cost. The evaluation metric is the aggregated weight of the selected edges, and the output should provide the declared total cost, the number of edges used, and the list of chosen edges.

    Task:
    - Implement the `solve` function that finds a minimum-cost Steiner tree connecting all terminal vertices.
    - The solution must only use edges present in the input graph.
    - The declared cost must match the sum of the selected edge weights.
    - All terminal vertices must be connected in the solution subgraph.
    - If any constraint is violated, the solution receives no score.

    Input arguments:
        instance_id  : (str) Unique identifier for this problem instance.
        n            : (int) Number of vertices in the graph.
        m            : (int) Number of edges in the graph.
        graph_edges  : (dict) Mapping of (min(u,v), max(u,v)) -> cost for each edge.
        terminals    : (list) List of terminal vertices (1-indexed).

    Returns:
        A dict with keys:
            - 'declared_cost': Total cost of the solution (a number).
            - 'num_edges': Number of edges in the solution (an integer).
            - 'edges': A list of tuples, each tuple (u, v) representing an edge in the solution.

    Key insights to explore:
    1. Shortest-path based heuristics (e.g., shortest path between terminals)
    2. Minimum spanning tree of the metric closure on terminals
    3. Pruning Steiner nodes that do not reduce cost
    4. Local search or metaheuristics (simulated annealing, genetic algorithms)
    5. Dynamic programming on subsets of terminals
    6. Integer programming formulations

    IMPORTANT: The main entry point is `def solve(instance_id, n, m, graph_edges, terminals)`.
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
