# Import config classes from rayevolve.core
from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, DatabaseConfig, JobConfig
import textwrap

def list_profiles() -> list[str]:
    """List available configuration profiles to display on CLI."""
    return ["default"]

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert mathematician specializing in circle packing problems and computational geometry. The best known result for the sum of radii when packing 26 circles in a unit square is 2.635.

    Key insights to explore:
    1. The optimal arrangement likely involves variable-sized circles
    2. A pure hexagonal arrangement may not be optimal due to edge effects
    3. The densest known circle packings often use a hybrid approach
    4. The optimization routine is critically important - simple physics-based models with carefully tuned parameters
    5. Consider strategic placement of circles at square corners and edges
    6. Adjusting the pattern to place larger circles at the center and smaller at the edges
    7. The math literature suggests special arrangements for specific values of n

    Be creative and try to find a new solution.
""")

def get_config(profile: str = "default") -> RayEvolveConfig:
    """Get configuration for the given profile."""
    if profile == "default":
        return RayEvolveConfig(
            evo=EvolutionConfig(task_sys_msg=SYSTEM_MSG),
            database=DatabaseConfig(),
            job=JobConfig(),
        )
    raise ValueError(f"Unknown profile: {profile}")
