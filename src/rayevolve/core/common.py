from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Dict, Any

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
        lang_identifier: Language used for any LLM code blocks (e.g. ```python ... ```). 
        evo_file: Name of the file to use for the evo block. Default is main.py.
        dl_evostate_freq: Frequency (in seconds) to download evo state from workers. 
    """
    results_dir: Optional[str] = None
    task_sys_msg: str =  ""
    num_agent_workers: int = 4
    max_generations: int = 50
    force_probing: bool = False 
    lang_identifier: str = "python" 
    evo_file: str = "main.py"
    dl_evostate_freq: float = 30


@dataclass(frozen=True)
class BackendConfig:
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
    backend: BackendConfig


def validate(cfg: RayEvolveConfig) -> None:
    """Validate the RayEvolveConfig object."""
    if cfg is None:
        raise ValueError("Config is None")
    for name in ("evo", "backend"):
        if not hasattr(cfg, name) or getattr(cfg, name) is None:
            raise ValueError(f"Config missing required section: {name}")

