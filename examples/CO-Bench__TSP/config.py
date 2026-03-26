"""Rayevolve config for CO-Bench Travelling Salesman Problem.

CO-Bench evaluation difference:
    The original CO-Bench benchmark enforces a short per-instance timeout
    (typically ~10s) via SIGALRM. Our evaluator uses a relaxed approach: all
    selected instances run in parallel as Ray remote tasks with a single global
    timeout (ray.wait, default 60s), and any that exceed it are cancelled.
"""

import os
import textwrap
from datetime import datetime
from pathlib import Path

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

    Input kwargs:
        instance_id : (str) Unique identifier for this problem instance, e.g. "tsp500_test_concorde_0".
        nodes       : (list of tuples) Each tuple is (x, y) coordinates.

    Returns:
        A dict with key "tour" containing a 0-indexed permutation of node indices.

    IMPORTANT: The main entry point is `def solve(**kwargs)`.
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
    gemini_pro = ModelSpec(
        description="Gemini 3 Pro Preview (thinking)",
        model=GoogleModel("gemini-3-pro-preview"),
        settings=GoogleModelSettings(google_thinking_config={"thinking_budget": 16384}),
    )
    # Gemini Flash 80%, Gemini Pro 20%
    return [gemini] * 4 + [gemini_pro] * 1


RAYEVOLVE_ROOT = Path(__file__).resolve().parent.parent.parent  # rayevolve/
RESULTS_DIR = RAYEVOLVE_ROOT / "results" / "CO-Bench__TSP"


def _make_results_dir(prefix: str = "") -> str:
    """Create a timestamped results dir under rayevolve/results/CO-Bench__TSP/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{timestamp}" if prefix else f"results_{timestamp}"
    return str(RESULTS_DIR / name)


def get_config(profile: str = "default") -> RayEvolveConfig:
    if profile == "default":
        return RayEvolveConfig(
            evo=EvolutionConfig(
                task_sys_msg=SYSTEM_MSG,
                build_strategy_model=build_strategy_model,
                build_evo_models=build_evo_models,
                results_dir=_make_results_dir("default"),
            ),
            backend=BackendConfig(),
        )
    if profile == "test":
        return RayEvolveConfig(
            evo=EvolutionConfig(
                task_sys_msg=SYSTEM_MSG,
                build_strategy_model=build_strategy_model,
                build_evo_models=build_evo_models,
                max_generations=6,
                num_agent_workers=6,
                results_dir=_make_results_dir("test"),
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
                results_dir=_make_results_dir("prod"),
            ),
            backend=BackendConfig(),
        )
    raise ValueError(f"Unknown profile: {profile}")
