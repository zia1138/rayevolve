"""Rayevolve config for CO-Bench 1D Bin Packing."""

import os
import textwrap

from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider


def list_profiles() -> list[str]:
    return ["default", "prod"]


SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in combinatorial optimization, specifically the one-dimensional
    bin packing problem. The goal is to minimize the number of fixed-capacity bins
    needed to pack all items.

    Problem specification:
    - Input: bin_capacity (float), num_items (int), items (list of float sizes)
    - Output: {'num_bins': int, 'bins': [[1-based item indices], ...]}
    - Items are 1-BASED indexed (first item is index 1, not 0)
    - Every item must appear exactly once across all bins
    - No bin may exceed bin_capacity
    - Score = best_known / num_bins (higher is better, max 1.0)

    Well-known approaches:
    1. First Fit Decreasing (FFD) - sort items descending, place each in first bin that fits
    2. Best Fit Decreasing (BFD) - sort descending, place in bin with least remaining space that fits
    3. Worst Fit Decreasing - sort descending, place in bin with most remaining space
    4. LP relaxation lower bounds + column generation
    5. Branch-and-price exact methods
    6. Local search: swap items between bins, merge under-filled bins
    7. Martello-Toth reduction procedures

    Key insights:
    - Sorting items descending before packing is almost always beneficial
    - Consider grouping items that sum close to bin_capacity
    - Hybrid strategies (greedy + local search refinement) often outperform pure greedy
    - The data includes instances with varying difficulty; focus on hard instances

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
