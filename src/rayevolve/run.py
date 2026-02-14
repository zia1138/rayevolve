#!/usr/bin/env python3
import os
import typer
import runpy
from pathlib import Path
from dataclasses import asdict
import ray

from rayevolve.core.runner import EvolutionRunner
from rayevolve.core.common import RayEvolveConfig
from rayevolve.core.common import validate

app = typer.Typer()


def load_config_file(config_file: Path) -> dict:
    if not config_file.exists():
        raise typer.BadParameter(f"Config file not found: {config_file}")
    return runpy.run_path(str(config_file))


def resolve_config_file(project: str) -> Path:
    """
    User passes a project/data folder. We assume it contains config.py.
    """
    p = Path(project)
    if not p.exists():
        raise typer.BadParameter(f"Project path not found: {project}")
    if not p.is_dir():
        raise typer.BadParameter(f"Project path must be a directory: {project}")

    config_file = p / "config.py"
    if not config_file.exists():
        raise typer.BadParameter(f"Expected config.py at: {config_file}")

    return config_file


DEFAULT_CONFIG_TEMPLATE = """\
# Import config classes from rayevolve.core
from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, DatabaseConfig, JobConfig

def list_profiles() -> list[str]:
    \"\"\"List available configuration profiles to display on CLI.\"\"\"
    return ["default"]

def get_config(profile: str = "default") -> RayEvolveConfig:
    \"\"\"Get configuration for the given profile.\"\"\"
    if profile == "default":
        return RayEvolveConfig(
            evo=EvolutionConfig(),
            database=DatabaseConfig(),
            job=JobConfig(),
        )
    raise ValueError(f"Unknown profile: {profile}")
"""


@app.command()
def init_config(
    project: str = typer.Argument(..., help="Project/data folder to create config.py in."),
    force: bool = typer.Option(False, help="Overwrite if config.py exists"),
):
    """
    Initialize a default config.py inside the given project folder.
    """
    project_path = Path(project)
    project_path.mkdir(parents=True, exist_ok=True)

    config_path = project_path / "config.py"
    if config_path.exists() and not force:
        raise typer.BadParameter(f"{config_path} already exists (use --force to overwrite)")

    config_path.write_text(DEFAULT_CONFIG_TEMPLATE)
    typer.echo(f"Created {config_path}")


@app.command()
def profiles(
    project: str = typer.Argument(..., help="Path to project/data folder containing config.py"),
):
    """
    List the profile availabes in the config.py of the given project folder.
    """
    config_file = resolve_config_file(project)
    ns = load_config_file(config_file)

    list_profiles = ns.get("list_profiles")
    if not callable(list_profiles):
        raise typer.BadParameter("config.py must define list_profiles()")
    for name in list_profiles():
        typer.echo(name)


@app.command()
def run(
    project_dir: str = typer.Argument(..., help="Path to project/data folder containing config.py"),
    profile: str = typer.Option("default"),
    seed: int = typer.Option(0),
    run_name: str = typer.Option("run"),
    dry_run: bool = typer.Option(False),
    ray_debug: bool = typer.Option(False, help="Enable Ray debug env vars (causes hangs on evolved script exceptions)"),
):
    """
    Run rayevolve using the given profile from config.py in the specified project directory.
    """
    config_file = resolve_config_file(project_dir)
    ns = load_config_file(config_file)

    get_config = ns.get("get_config")
    if not callable(get_config):
        raise typer.BadParameter("config.py must define get_config(profile)")

    cfg: RayEvolveConfig = get_config(profile)
    validate(cfg)

    typer.echo(f"run_name={run_name} seed={seed} profile={profile}")
    typer.echo(asdict(cfg))

    if dry_run:
        return

    # Forward API keys to Ray workers.  Ray spawns worker processes that do
    # NOT inherit the parent's environment variables.  pydantic-ai's
    # GoogleModel reads GOOGLE_API_KEY from os.environ inside each worker,
    # so without explicit forwarding every agent call fails with an auth
    # error.  runtime_env["env_vars"] injects these into every worker.
    env_vars = {}
    for key in ("GOOGLE_API_KEY",):
        val = os.environ.get(key)
        if val:
            env_vars[key] = val
    if ray_debug:
        env_vars["RAY_DEBUG"] = "1"
        env_vars["RAY_DEBUG_POST_MORTEM"] = "1"

    runtime_env = {"env_vars": env_vars} if env_vars else None
    ray.init(runtime_env=runtime_env)

    try:
        # TODO: Might want to use the RayEvolveConfig eventually.
        evo_runner = EvolutionRunner(cfg.evo, cfg.job, cfg.database, project_dir, verbose=True)
        evo_runner.run_ray()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    app()