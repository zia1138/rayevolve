"""Rayevolve config for CO-Bench Travelling Salesman Problem."""

import os
import textwrap

from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider


def list_profiles() -> list[str]:
    return ["default", "prod"]


SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in combinatorial optimization, specifically the Euclidean
    Travelling Salesman Problem (TSP). The goal is to find the shortest tour
    visiting all cities exactly once and returning to the start.

    Problem specification:
    - Input: nodes (list of (x, y) coordinate tuples)
    - Output: {'tour': [0-indexed permutation of node indices]}
    - Nodes are 0-BASED indexed
    - Tour is a closed cycle (last node connects back to first)
    - Distance is Euclidean: sqrt((x2-x1)^2 + (y2-y1)^2)
    - Score = optimal_cost / predicted_cost (higher is better, max 1.0)

    Well-known approaches:
    1. Nearest Neighbor - greedy, start from each node and pick best
    2. 2-opt - iteratively reverse tour segments to reduce crossings
    3. 3-opt - consider 3-edge exchanges
    4. Or-opt - move sequences of 1-3 cities to better positions
    5. Lin-Kernighan style variable-depth search
    6. Christofides algorithm (1.5-approximation guarantee)
    7. Greedy edge insertion + local search
    8. Simulated annealing with 2-opt moves

    Key insights:
    - Nearest neighbor is a weak baseline; 2-opt almost always improves it significantly
    - Multiple random starting points + 2-opt is a strong simple strategy
    - For 500-1000 nodes, 2-opt runs quickly and gets within 5-10% of optimal
    - Eliminating crossing edges is the single most impactful improvement
    - Consider spatial data structures (k-d trees) for nearest neighbor lookups
    - Instance sizes: 500 and 1000 cities (must complete within timeout)

    NOTE: solve(**kwargs) is the main entry point of the code.
""")


def build_strategy_model() -> ModelSpec:
    return ModelSpec(
        description="GEMINI 3 Flash Preview",
        model=GoogleModel("gemini-3-flash-preview"),
        settings=GoogleModelSettings(),
    )


def build_evo_models() -> list[ModelSpec]:
    gemini = ModelSpec(
        description="Gemini 3 Flash Preview (thinking)",
        model=GoogleModel("gemini-3-flash-preview"),
        settings=GoogleModelSettings(google_thinking_config={"thinking_budget": 8192}),
    )
    sonnet = ModelSpec(
        description="Claude Sonnet 4.6 via Lightning AI",
        model=OpenAIChatModel(
            "anthropic/claude-sonnet-4-6",
            provider=OpenAIProvider(
                base_url="https://lightning.ai/api/v1/",
                api_key=os.environ["LIGHTNING_API_KEY"] + "/delfidiagnostics",
            ),
        ),
        settings=OpenAIChatModelSettings(),
    )
    # Gemini Flash 80%, Claude Sonnet 20%
    return [gemini] * 4 + [sonnet] * 1


def get_config(profile: str = "default") -> RayEvolveConfig:
    if profile == "default":
        return RayEvolveConfig(
            evo=EvolutionConfig(
                task_sys_msg=SYSTEM_MSG,
                build_strategy_model=build_strategy_model,
                build_evo_models=build_evo_models,
            ),
            backend=BackendConfig(),
        )
    if profile == "prod":
        return RayEvolveConfig(
            evo=EvolutionConfig(
                task_sys_msg=SYSTEM_MSG,
                build_strategy_model=build_strategy_model,
                build_evo_models=build_evo_models,
                max_generations=60,
                num_agent_workers=12,
            ),
            backend=BackendConfig(),
        )
    raise ValueError(f"Unknown profile: {profile}")
