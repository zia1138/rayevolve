# Import config classes from rayevolve.core
from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, DatabaseConfig, JobConfig
import textwrap

SYSTEM_MSG = textwrap.dedent("""\
    You are given a tabular dataset and a supervised prediction and feature engineering task. Your objective is to
    maximize true positive rate (TPR) at a false positive rate (FPR) of 0.05 on a held-out validation set by learning an
    effective set of features and a predictive model. Achieving strong performance requires explicitly discovering and
    reasoning about relationships in the data, including cases where the effect of one variable depends on another,
    where subgroups behave differently, or where combinations of variables carry information not present in any single
    feature. You should inspect the training data from multiple perspectives (e.g., feature distributions, associations
    between variables, conditional behavior, and subgroup statistics), form concrete hypotheses about data structure,
    and modify the feature representation accordingly through transformations, encodings, or constructed feature
    combinations. Since evaluation depends only on performance at FPR <= 0.05, feature choices should prioritize
    improving discrimination among the highest-confidence predictions, even if they do not improve overall ranking
    metrics. You should also analyze model confidence and error structure around this operating point, including
    high-confidence false positives, missed positives near the decision boundary, and score distributions within
    important subgroups, and use these findings to motivate further feature engineering. You may also modify modeling
    choices, hyperparameters, and decision rules, but these should support - not replace - effective feature engineering. No
    domain-specific assumptions or predefined feature relationships are provided; all useful structure must be
    discovered empirically from the data. Evaluation is performed by a fixed external evaluator reporting TPR at FPR =
    0.05 on validation data. You do not have access to validation labels.
""")

def list_profiles() -> list[str]:
    """List available configuration profiles to display on CLI."""
    return ["default"]

def get_config(profile: str = "default") -> RayEvolveConfig:
    """Get configuration for the given profile."""
    if profile == "default":
        return RayEvolveConfig(
            evo=EvolutionConfig(task_sys_msg=SYSTEM_MSG),
            database=DatabaseConfig(),
            job=JobConfig(),
        )
    raise ValueError(f"Unknown profile: {profile}")
