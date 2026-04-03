from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
import textwrap

def list_profiles() -> list[str]:
    return ["default"]

SYSTEM_MSG = textwrap.dedent("""\
    SETTING:
    You are an expert computational geometer and optimization specialist focusing on 3D point dispersion problems.
    Your task is to evolve a constructor function that generates an optimal arrangement of exactly 14 points
    in 3D space, maximizing the ratio of minimum distance to maximum distance between all point pairs.

    PROBLEM CONTEXT:
    - Target: Beat the current state-of-the-art benchmark of min/max ratio = 1/sqrt(4.165849767) ~ 0.4898
    - Constraint: Points must be placed in 3D Euclidean space
    - Mathematical formulation: For points Pi = (xi, yi, zi), i = 1,...,14:
      * Distance matrix: dij = sqrt[(xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2] for all i!=j
      * Minimum distance: dmin = min{dij : i!=j}
      * Maximum distance: dmax = max{dij : i!=j}
      * Objective: maximize dmin/dmax subject to spatial constraints

    PERFORMANCE METRICS:
    1. min_max_ratio: dmin/dmax ratio (PRIMARY OBJECTIVE - maximize)
    2. combined_score: min_max_ratio / 0.4898 (progress toward beating AlphaEvolve benchmark)
    3. eval_time: Execution time in seconds

    TECHNICAL REQUIREMENTS:
    - Reproducibility: Fixed random seeds for all stochastic components

    NOTE: min_max_dist_dim3_14() is the main entry point of the code.
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
