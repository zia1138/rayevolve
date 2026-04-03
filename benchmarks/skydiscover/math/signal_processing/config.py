from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
import textwrap

def list_profiles() -> list[str]:
    return ["default"]

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert signal processing engineer specializing in real-time adaptive filtering algorithms. Your
    task is to improve a signal processing algorithm that filters volatile, non-stationary time series data using a sliding
    window approach. The algorithm must minimize noise while preserving signal dynamics with minimal computational latency
    and phase delay. Focus on the multi-objective optimization of: (1) Slope change minimization - reducing spurious directional
    reversals, (2) Lag error minimization - maintaining responsiveness, (3) Tracking accuracy - preserving genuine signal
    trends, and (4) False reversal penalty - avoiding noise-induced trend changes. Consider advanced techniques like adaptive
    filtering (Kalman filters, particle filters), multi-scale processing (wavelets, EMD), predictive enhancement (polynomial
    fitting, neural networks), and trend detection methods.

    NOTE: run_signal_processing() is the main entry point of the code.
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
