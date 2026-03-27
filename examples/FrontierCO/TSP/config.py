from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

import textwrap

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in combinatorial optimization, specifically the Traveling Salesman Problem.

    Problem Description:
    The Traveling Salesman Problem (TSP) is a classic combinatorial optimization problem where, given a set of cities with known pairwise distances, the objective is to find the shortest possible tour that visits each city exactly once and returns to the starting city. More formally, given a complete graph G = (V, E) with vertices V representing cities and edges E with weights representing distances, we seek to find a Hamiltonian cycle (a closed path visiting each vertex exactly once) of minimum total weight.

    Task:
    - Implement the `solve` function that finds the shortest tour visiting all cities exactly once.
    - The tour must visit each city exactly once and return to the starting city.
    - The tour is represented as a list of node indices.
    - The objective is to minimize the total Euclidean distance of the tour.

    Input arguments:
        instance_id : (str) Unique identifier for this problem instance.
        nodes       : (list) List of (x, y) coordinates representing cities.
                      Format: [(x1, y1), (x2, y2), ..., (xn, yn)]

    Returns:
        A dict with key:
            - 'tour': List of node indices representing the solution path.
                      Format: [0, 3, 1, ...] where numbers are indices into the nodes list.

    Key insights to explore:
    1. Nearest neighbor heuristic
    2. 2-opt and 3-opt local search improvements
    3. Christofides algorithm
    4. Lin-Kernighan heuristic
    5. Simulated annealing or genetic algorithms
    6. Branch and bound for exact solutions on small instances

    IMPORTANT: The main entry point is `def solve(instance_id, nodes)`.
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
