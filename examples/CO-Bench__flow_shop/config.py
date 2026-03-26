"""Rayevolve config for CO-Bench Flow Shop Scheduling Problem.

CO-Bench evaluation difference:
    The original CO-Bench benchmark enforces a strict 60-second per-instance
    timeout via SIGALRM. Our evaluator instead runs all selected instances in
    parallel as Ray remote tasks with a single global timeout (ray.wait), and
    cancels any that exceed it. This "relaxed" approach avoids signal-based
    constraints and lets Ray manage concurrency, but means a slow instance
    only costs wall-clock time rather than being hard-killed at exactly 60s.
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
    You are an expert in combinatorial optimization, specifically the permutation
    flow shop scheduling problem. The goal is to find a job ordering that minimizes
    the makespan (total completion time of all jobs on all machines).

    Problem Description:
    Given n jobs and m machines, each job must be processed on every machine in
    the fixed order machine 0, machine 1, ..., machine m-1. Each machine processes
    one job at a time and no preemption is allowed. The task is to find a permutation
    of jobs that minimizes the makespan — the time at which the last job completes
    on the last machine. The makespan is computed via the classical recurrence:
      C[i][j] = max(C[i-1][j], C[i][j-1]) + processing_time(job_i, machine_j)
    The score is lower_bound / makespan (higher is better, max 1.0).

    Input kwargs:
        instance_id  : (str) Unique identifier, e.g. "tai20_5_0".
        n            : (int) Number of jobs.
        m            : (int) Number of machines.
        matrix       : (list of lists) matrix[job][machine] processing times.
        upper_bound  : (int) Known upper bound for this instance.
        lower_bound  : (int) Known lower bound for this instance.

    Returns:
        A dict with key "job_sequence" containing a 1-indexed permutation of jobs,
        e.g. {'job_sequence': [3, 1, 2, ...]} for 3 jobs.

    Well-known approaches:
    1. NEH (Nawaz-Enscore-Ham) - sort by total processing time, insertion heuristic
    2. CDS (Campbell-Dudek-Smith) - apply Johnson's rule on machine subsets
    3. Palmer's slope index
    4. Simulated annealing with adjacent swap neighborhood
    5. Genetic algorithms with order crossover (OX)
    6. Iterated local search with perturbation
    7. Beam search with makespan-based evaluation

    Key insights:
    - NEH is a strong baseline; improvements usually come from local search refinement
    - Adjacent-swap and insert neighborhoods are most effective for flow shop
    - Tiebreaking in NEH (when two positions give equal makespan) matters significantly
    - For large instances (200+ jobs), population-based or restart strategies help
    - Instance sizes range from 20x5 to 500x20 (must complete within timeout)
    - The Taillard benchmarks have well-studied upper/lower bounds

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
RESULTS_DIR = RAYEVOLVE_ROOT / "results" / "CO-Bench__flow_shop"


def _make_results_dir(prefix: str = "") -> str:
    """Create a timestamped results dir under rayevolve/results/CO-Bench__flow_shop/."""
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
