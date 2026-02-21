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


def _validate_port(ctx: typer.Context, param: typer.CallbackParam, value: int) -> int:
    if not (1 <= value <= 65535):
        raise typer.BadParameter("Port must be in range 1–65535")
    return value


def _normalize_nonempty(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    return s or None


@app.command()
def run(
    project_dir: str = typer.Argument(..., help="Path to project/data folder containing config.py"),
    profile: str = typer.Option("default"),
    seed: int = typer.Option(0),
    run_name: str = typer.Option("run"),
    dry_run: bool = typer.Option(False),
    ray_debug: bool = typer.Option(False, help="Enable Ray debug env vars (RAY_DEBUG, RAY_DEBUG_POST_MORTEM)"),
    ray_address: str | None = typer.Option(
        None,
        help=(
            "Optional Ray Client address (ray://<host>:<port>). "
            "If not provided, a local Ray runtime is started."
        ),
    ),
    ray_ip: str | None = typer.Option(
        None,
        help="Alternative to --ray-address: specify Ray head IP/host.",
    ),
    ray_port: int = typer.Option(
        10001,
        callback=_validate_port,
        help="Ray Client port (default: 10001). Only used with --ray-ip.",
        show_default=True,
    ),
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

    env_vars: dict[str, str] = {}
    if ray_debug:
        env_vars["RAY_DEBUG"] = "1"
        env_vars["RAY_DEBUG_POST_MORTEM"] = "1"  # fixed typo

    runtime_env = {"env_vars": env_vars} if env_vars else None

    ray_address = _normalize_nonempty(ray_address)
    ray_ip = _normalize_nonempty(ray_ip)

    if ray_address and ray_ip:
        raise typer.BadParameter("Use either --ray-address or --ray-ip/--ray-port, not both.")

    # Initialize Ray
    if ray_address:
        if not ray_address.startswith("ray://"):
            raise typer.BadParameter(
                "Ray address must start with ray://, e.g. ray://127.0.0.1:10001"
            )
        ray.init(address=ray_address, runtime_env=runtime_env)

    elif ray_ip:
        ray.init(address=f"ray://{ray_ip}:{ray_port}", runtime_env=runtime_env)

    else:
        ray.init(runtime_env=runtime_env)

    try:
        evo_runner = EvolutionRunner(
            cfg.evo,
            cfg.job,
            cfg.database,
            project_dir,
            verbose=True,
        )
        evo_runner.run_ray()

    finally:
        ray.shutdown()


if __name__ == "__main__":
    app()