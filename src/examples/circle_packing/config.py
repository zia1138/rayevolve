# Import config classes from rayevolve.core
from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, DatabaseConfig, JobConfig

def list_profiles() -> list[str]:
    """List available configuration profiles to display on CLI."""
    return ["default"]

def get_config(profile: str = "default") -> RayEvolveConfig:
    """Get configuration for the given profile."""
    if profile == "default":
        return RayEvolveConfig(
            evo=EvolutionConfig(),
            database=DatabaseConfig(),
            job=JobConfig(),
        )
    raise ValueError(f"Unknown profile: {profile}")
