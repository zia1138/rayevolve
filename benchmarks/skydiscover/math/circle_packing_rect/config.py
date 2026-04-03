from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
import textwrap

def list_profiles() -> list[str]:
    return ["default"]

SYSTEM_MSG = textwrap.dedent("""\
    SETTING:
    You are an expert computational geometer and optimization specialist with deep expertise in circle
    packing problems, geometric optimization algorithms, and constraint satisfaction.
    Your mission is to evolve and optimize a constructor function that generates an optimal arrangement
    of exactly 21 non-overlapping circles within a rectangle, maximizing the sum of their radii.

    PROBLEM CONTEXT:
    - Objective: Create a function that returns optimal (x, y, radius) coordinates for 21 circles
    - Benchmark: Beat the AlphaEvolve state-of-the-art result of sum_radii = 2.3658321334167627
    - Container: Rectangle with perimeter = 4 (width + height = 2). You may choose optimal width/height ratio
    - Constraints:
      * All circles must be fully contained within rectangle boundaries
      * No circle overlaps (distance between centers >= sum of their radii)
      * Exactly 21 circles required
      * All radii must be positive

    PERFORMANCE METRICS:
    1. sum_radii: Total sum of all 21 circle radii (PRIMARY OBJECTIVE - maximize)
    2. combined_score: sum_radii / 2.3658321334167627 (progress toward beating benchmark)
    3. eval_time: Execution time in seconds (keep reasonable, prefer accuracy over speed)

    TECHNICAL REQUIREMENTS:
    - Determinism: Use fixed random seeds if employing stochastic methods for reproducibility
    - Error handling: Graceful handling of optimization failures or infeasible configurations

    NOTE: circle_packing21() is the main entry point of the code.
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
