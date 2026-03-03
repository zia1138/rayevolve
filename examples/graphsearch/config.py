# Import config classes from rayevolve.core
from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
import textwrap

SYSTEM_MSG = textwrap.dedent("""\
    You are designing a graph-search algorithm.

    Task:
    - Implement the `graph_search` function.
    - Given a graph with a start node and a goal node, return a valid path if one exists.
    - You may interact with the graph only by requesting the neighbors of a node and checking whether a node is the goal via the provided `env` object.

    Evaluation:
    - Your `graph_search` function will be called with an `env` instance.
    - The solution is invalid if it fails to find a path on any instance.
    - Cost is proportional to:
      - the total number of neighbor queries,
      - and the length of the returned path.
    - Lower total cost is better, subject to full correctness.

    Note:
    - Some graphs contain hidden structural constraints that make locally optimal decisions misleading.
    - Lightweight probing to infer structure from interaction may help reduce overall cost.

    IMPORTANT IMPLEMENTATION DETAILS:
    - Do NOT modify the `SearchEnv` class definition provided in the template. The evaluator will pass a compatible subclass instance at runtime.
    - Your code must define `graph_search(env) -> Optional[List]`.
                             
    The main entry point of this code is: `def graph_search(env: SearchEnv) -> Optional[List]`
""")

def list_profiles() -> list[str]:
    """List available configuration profiles to display on CLI."""
    return ["default"]


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
    """Get configuration for the given profile."""
    if profile == "default":
        return RayEvolveConfig(
            evo=EvolutionConfig(task_sys_msg=SYSTEM_MSG,
                                build_strategy_model=build_strategy_model,
                                build_evo_models=build_evo_models),
            backend=BackendConfig(),
        )
    raise ValueError(f"Unknown profile: {profile}")
