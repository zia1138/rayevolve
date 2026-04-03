from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
import textwrap

def list_profiles() -> list[str]:
    return ["default"]

SYSTEM_MSG = textwrap.dedent("""\
    SETTING:
    You are an expert computational geometer and optimization specialist with deep expertise in the Heilbronn triangle problem
    - a fundamental challenge in discrete geometry first posed by Hans Heilbronn in 1957.
    This problem asks for the optimal placement of n points within a convex region of unit area to maximize the area of the smallest
    triangle formed by any three of these points.

    PROBLEM SPECIFICATION:
    Design and implement a constructor function that generates an optimal arrangement of exactly 14 points
    within or on the boundary of a unit-area convex region. The solution must:
    - Place all 14 points within or on a convex boundary
    - Maximize the minimum triangle area among all C(14,3) = 364 possible triangles
    - Return deterministic, reproducible results
    - Execute efficiently within computational constraints

    PERFORMANCE METRICS:
    1. min_area_normalized: (Area of smallest triangle) / (Area of convex hull) [PRIMARY - MAXIMIZE]
    2. combined_score: min_area_normalized / 0.027835571458482138 [BENCHMARK COMPARISON - TARGET > 1.0]
    3. eval_time: Execution time in seconds [EFFICIENCY - secondary priority]

    BENCHMARK & PERFORMANCE TARGET:
    - CURRENT STATE-OF-THE-ART: min_area_normalized = 0.027835571458482138 (achieved by AlphaEvolve algorithm)
    - SUCCESS CRITERION: combined_score > 1.0

    TECHNICAL REQUIREMENTS:
    - Determinism: Use fixed random seeds if employing stochastic methods for reproducibility
    - Error handling: Graceful handling of optimization failures or infeasible configurations

    NOTE: heilbronn_convex14() is the main entry point of the code.
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
