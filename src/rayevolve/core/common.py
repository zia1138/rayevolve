from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Dict, Any

FOLDER_PREFIX = "gen"

@dataclass(frozen=True)
class EvolutionConfig:
    """
    Configuration for rayevolve run.

    Attributes:
        results_dir: Optional path to save results. If None, a timestamped folder will be created.
        task_sys_msg: Optional system message for the task.
        num_agent_workers: Number of agent workers to use.
        max_generations: Maximum number of program generations to evolve.
        force_probing: Whether to force probing of evo block during multi-turn loop for EvoExplore/EvoExploit.
    """
    results_dir: Optional[str] = None
    task_sys_msg: str =  ""
    num_agent_workers: int = 4
    max_generations: int = 50
    force_probing: bool = False 


class DatabaseConfig:
    """Configuration for program database. ShinkEvolve code still needs to be
       removed from this component."""
    num_islands: int = 1
    archive_size: int = 100
    elite_selection_ratio: float = 0.3  # Prop of elites inspirations
    num_archive_inspirations: int = 5  # No. inspiration programs
    num_top_k_inspirations: int = 2  # No. top-k inspiration programs
    migration_interval: int = 10  # Migrate every N generations
    migration_rate: float = 0.1  # Prop. of island pop. to migrate
    island_elitism: bool = True  # Keep best prog on their islands
    enforce_island_separation: bool = (
        True  # Enforce full island separation for inspirations
    )
    parent_selection_strategy: str = (
        "power_law"  # "weighted"/"power_law" / "beam_search"
    )
    exploitation_alpha: float = 1.0  # 0=uniform, 1=power-law
    exploitation_ratio: float = 0.2  # Chance to pick from archive
    parent_selection_lambda: float = 10.0  # >0 sharpness of sigmoid
    num_beams: int = 5
    embedding_model: str = "text-embedding-3-small"

@dataclass(frozen=True)
class JobConfig:
    """
    Configuration for script execution.

    Attributes:
        timeout_sec: Optional timeout in seconds for script execution. If None, no timeout is applied
        extra_cmd_args: Optional dictionary of extra command-line arguments to pass to the evaluation script.
        conda_env: Optional name of the conda environment to use for execution. If None, the current environment is used.
    """
    extra_cmd_args: Dict[str, Any] = field(default_factory=dict)
    timeout_sec: int = 10 * 60
    conda_env: Optional[str] = None

@dataclass(frozen=True)
class RayEvolveConfig:
    evo: EvolutionConfig
    database: DatabaseConfig
    job: JobConfig


def validate(cfg: RayEvolveConfig) -> None:
    """Validate the RayEvolveConfig object."""
    if cfg is None:
        raise ValueError("Config is None")
    for name in ("evo", "database", "job"):
        if not hasattr(cfg, name) or getattr(cfg, name) is None:
            raise ValueError(f"Config missing required section: {name}")

