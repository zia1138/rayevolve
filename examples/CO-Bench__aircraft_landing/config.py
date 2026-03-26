"""Rayevolve config for CO-Bench Aircraft Landing Scheduling Problem.

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
    You are an expert in combinatorial optimization, specifically the Aircraft Landing Scheduling Problem.

    Problem Description:
    The problem is to schedule landing times for a set of planes across one or more runways such that each landing occurs within its prescribed time window and all pairwise separation requirements are satisfied; specifically, if plane i lands at or before plane j on the same runway, then the gap between their landing times must be at least the specified separation time provided in the input. In a multiple-runway setting, each plane must also be assigned to one runway, and if planes land on different runways, the separation requirement (which may differ) is applied accordingly. Each plane has an earliest, target, and latest landing time, with penalties incurred proportionally for landing before (earliness) or after (lateness) its target time. The objective is to minimize the total penalty cost while ensuring that no constraints are violated—if any constraint is breached, the solution receives no score.

    Task:
    - Implement the `solve` function that schedules landing times and assigns runways for a set of planes.
    - Each plane has an earliest, target, and latest landing time, with penalties for deviating from the target.
    - If plane i lands at or before plane j on the same runway, the gap between their landing times must be
      at least the specified separation time.
    - The objective is to minimize the total penalty cost while satisfying all constraints.
    - If any constraint is violated, the solution receives no score.

    Input kwargs:
        instance_id : (str) Unique identifier for this problem instance, e.g. "airland1_0".
        num_planes  : (int) Number of planes.
        num_runways : (int) Number of runways.
        freeze_time : (float) Freeze time.
        planes      : (list of dict) Each with keys: "appearance", "earliest", "target", "latest",
                      "penalty_early", "penalty_late".
        separation  : (list of lists) separation[i][j] is the required gap after plane i lands before
                      plane j can land on the same runway.

    Returns:
        A dict with key "schedule" mapping each plane id (1-indexed) to a dict with
        "landing_time" (float) and "runway" (int).

    Key insights to explore:
    1. Greedy scheduling sorted by target or earliest time
    2. Constraint propagation to tighten time windows
    3. Distributing planes across runways to reduce separation conflicts
    4. Local search or metaheuristics (simulated annealing, genetic algorithms)
    5. Mixed-integer programming formulations
    6. Priority-based heuristics using penalty costs to break ties

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
RESULTS_DIR = RAYEVOLVE_ROOT / "results" / "CO-Bench__aircraft_landing"


def _make_results_dir(prefix: str = "") -> str:
    """Create a timestamped results dir under rayevolve/results/CO-Bench__aircraft_landing/."""
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
