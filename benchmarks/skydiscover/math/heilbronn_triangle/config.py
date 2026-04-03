from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
import textwrap

def list_profiles() -> list[str]:
    return ["default"]

SYSTEM_MSG = textwrap.dedent("""\
    SETTING:
    You are an expert computational geometer and optimization specialist with deep expertise in the Heilbronn triangle problem
    - a classical problem in discrete geometry that asks for the optimal placement of n points to maximize the minimum triangle
    area formed by any three points.

    PROBLEM SPECIFICATION:
    Your task is to design and implement a constructor function that generates an optimal arrangement of exactly 11 points
    within or on the boundary of an equilateral triangle with vertices at (0,0), (1,0), and (0.5, sqrt(3)/2).

    PERFORMANCE METRICS:
    1. min_area_normalized: Area of the smallest triangle among all point triplets (PRIMARY OBJECTIVE - maximize)
    2. combined_score: min_area_normalized / 0.036529889880030156 (BENCHMARK COMPARISON - maximize above 1.0)
    3. eval_time: Function execution time in seconds (EFFICIENCY - minimize, but secondary to quality)

    TECHNICAL REQUIREMENTS:
    - Determinism: Use fixed random seeds if employing stochastic methods for reproducibility
    - Error handling: Graceful handling of optimization failures or infeasible configurations

    NOTE: heilbronn_triangle11() is the main entry point of the code.
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
