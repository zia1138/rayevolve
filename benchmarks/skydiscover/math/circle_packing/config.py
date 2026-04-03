from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
import textwrap

def list_profiles() -> list[str]:
    return ["default"]

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert mathematician specializing in circle packing problems and computational geometry. Your
    task is to improve a constructor function that directly produces a specific arrangement of 26 circles in a unit square,
    maximizing the sum of their radii. The AlphaEvolve paper achieved a sum of 2.635 for n=26.

    Key geometric insights:
    - Circle packings often follow hexagonal patterns in the densest regions
    - Maximum density for infinite circle packing is pi/(2*sqrt(3)) ≈ 0.9069
    - Edge effects make square container packing harder than infinite packing
    - Circles can be placed in layers or shells when confined to a square
    - Similar radius circles often form regular patterns, while varied radii allow better space utilization
    - Perfect symmetry may not yield the optimal packing due to edge effects

    Focus on designing an explicit constructor that places each circle in a specific position, rather than an iterative search
    algorithm.

    NOTE: run_packing() is the main entry point of the code.
""")


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
    if profile == "default":
        return RayEvolveConfig(
            evo=EvolutionConfig(task_sys_msg=SYSTEM_MSG,
                                build_strategy_model=build_strategy_model,
                                build_evo_models=build_evo_models),
            backend=BackendConfig(),
        )
    raise ValueError(f"Unknown profile: {profile}")
