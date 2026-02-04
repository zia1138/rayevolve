from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Dict, Any

FOLDER_PREFIX = "gen"

@dataclass(frozen=True)
class EvolutionConfig:
    results_dir: Optional[str] = None
    task_sys_msg: Optional[str] = None
    num_generations: int = 10
    max_parallel_jobs: int = 2
    max_patch_resamples: int = 3
    max_patch_attempts: int = 5
    job_type: str = "local"
    language: str = "python"
    llm_models: List[str] = field(default_factory=lambda: ["azure-gpt-4.1-mini"])
    llm_dynamic_selection: Optional[str] = None
    llm_dynamic_selection_kwargs: dict = field(default_factory=lambda: {})
    llm_kwargs: dict = field(default_factory=lambda: {})
    meta_rec_interval: Optional[int] = None
    meta_llm_models: Optional[List[str]] = None
    meta_llm_kwargs: dict = field(default_factory=lambda: {})
    meta_max_recommendations: int = 5
    embedding_model: Optional[str] = None
    max_novelty_attempts: int = 3
    code_embed_sim_threshold: float = 1.0
    novelty_llm_models: Optional[List[str]] = None
    novelty_llm_kwargs: dict = field(default_factory=lambda: {})
    use_text_feedback: bool = False


class DatabaseConfig:
    num_islands: int = 4
    archive_size: int = 100

    # Inspiration parameters
    elite_selection_ratio: float = 0.3  # Prop of elites inspirations
    num_archive_inspirations: int = 5  # No. inspiration programs
    num_top_k_inspirations: int = 2  # No. top-k inspiration programs

    # Island model/migration parameters
    migration_interval: int = 10  # Migrate every N generations
    migration_rate: float = 0.1  # Prop. of island pop. to migrate
    island_elitism: bool = True  # Keep best prog on their islands
    enforce_island_separation: bool = (
        True  # Enforce full island separation for inspirations
    )

    # Parent selection parameters
    parent_selection_strategy: str = (
        "power_law"  # "weighted"/"power_law" / "beam_search"
    )

    # Power-law parent selection parameters
    exploitation_alpha: float = 1.0  # 0=uniform, 1=power-law
    exploitation_ratio: float = 0.2  # Chance to pick from archive

    # Weighted tree parent selection parameters
    parent_selection_lambda: float = 10.0  # >0 sharpness of sigmoid

    # Beam search parent selection parameters
    num_beams: int = 5

    # Embedding model name
    embedding_model: str = "text-embedding-3-small"

@dataclass(frozen=True)
class JobConfig:
    eval_program_path: Optional[str] = "evaluate.py"
    extra_cmd_args: Dict[str, Any] = field(default_factory=dict)
    time: Optional[str] = None
    conda_env: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        job_to_dict = asdict(self)
        return {k: v for k, v in job_to_dict.items() if v is not None}


@dataclass(frozen=True)
class RayEvolveConfig:
    evo: EvolutionConfig
    database: DatabaseConfig
    job: JobConfig



