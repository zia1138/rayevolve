"""Rayevolve config for CO-Bench Flow Shop Scheduling."""

import os
import textwrap

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

    Problem specification:
    - Input: n (jobs), m (machines), matrix (n x m processing times)
    - Output: {'job_sequence': [1-indexed permutation of jobs]}
    - Jobs are 1-BASED indexed (first job is 1, not 0)
    - Each job visits machines 0..m-1 in order
    - Makespan computed via classical recurrence:
      C[i][j] = max(C[i-1][j], C[i][j-1]) + processing_time(job_i, machine_j)
    - Score = lower_bound / makespan (higher is better, max 1.0)

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
    - The Taillard benchmarks have well-studied upper/lower bounds

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
