# Import config classes from rayevolve.core
from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig
import textwrap

SYSTEM_MSG = textwrap.dedent("""\
    You are given a tabular dataset (UCI Adult) and a supervised prediction task. 
    Your objective is to maximize the True Positive Rate (TPR) at a fixed False Positive Rate (FPR) of 0.05 on the validation set.

    Task:
    - Implement the `preprocess_train_and_predict` function.
    - Perform feature engineering and train a model on (X_train, y_train).
    - Return predicted probabilities for the positive class (1) for X_val.

    Evaluation:
    - Performance is measured solely by TPR at FPR = 0.05.
    - You must return a DataFrame with a "y_proba" column aligned with the X_val index.

    Be creative with feature engineering (encodings, transformations, combinations) and model selection.
""")

def list_profiles() -> list[str]:
    """List available configuration profiles to display on CLI."""
    return ["default"]

def get_config(profile: str = "default") -> RayEvolveConfig:
    """Get configuration for the given profile."""
    if profile == "default":
        return RayEvolveConfig(
            evo=EvolutionConfig(task_sys_msg=SYSTEM_MSG),
            backend=BackendConfig(),
        )
    raise ValueError(f"Unknown profile: {profile}")
