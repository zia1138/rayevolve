#!/usr/bin/env python3
import os
import contextlib
import concurrent.futures
import typer
import runpy
from typing import List
from pathlib import Path
from dataclasses import asdict
import ray

from rayevolve.core.runner import EvolutionRunner
from rayevolve.core.common import RayEvolveConfig
from rayevolve.core.common import validate

app = typer.Typer()


def load_config_file(config_file: Path) -> dict:
    """
    We use a code as config system so runpy runs config.py to populate the namespace.
    We expect config.py to define at least get_config(profile) and list_profiles() functions. 
    The rest is up to the user.
    """
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
from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
import textwrap

def list_profiles() -> list[str]:
    \"\"\"List available configuration profiles to display on CLI.\"\"\"
    return ["default"]

SYSTEM_MSG = textwrap.dedent(\"\"\"\\
    You are an expert in <DOMAIN>. Describe the task and any key insights here.

    NOTE: <entry_function>() is the main entry point of the code.
\"\"\")


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
    \"\"\"Get configuration for the given profile.\"\"\"
    if profile == "default":
        return RayEvolveConfig(
            evo=EvolutionConfig(
                task_sys_msg=SYSTEM_MSG,
                build_strategy_model=build_strategy_model,
                build_evo_models=build_evo_models,
            ),
            backend=BackendConfig(),
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


@contextlib.contextmanager
def _ray_session(ray_address: str | None, ray_ip: str | None, ray_port: int, ray_debug: bool):
    """Initialize Ray, yield, then shut down."""
    env_vars: dict[str, str] = {}
    if ray_debug:
        env_vars["RAY_DEBUG"] = "1"
        env_vars["RAY_DEBUG_POST_MORTEM"] = "1"

    runtime_env = {"env_vars": env_vars} if env_vars else None

    ray_address = _normalize_nonempty(ray_address)
    ray_ip = _normalize_nonempty(ray_ip)

    if ray_address and ray_ip:
        raise typer.BadParameter("Use either --ray-address or --ray-ip/--ray-port, not both.")

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
        yield
    finally:
        ray.shutdown()


def _run_single_project(project_dir: str, profile: str) -> None:
    """Load config and run evolution for a single project."""
    config_file = resolve_config_file(project_dir)
    ns = load_config_file(config_file)

    get_config = ns.get("get_config")
    if not callable(get_config):
        raise typer.BadParameter(f"config.py in {project_dir} must define get_config(profile)")

    cfg: RayEvolveConfig = get_config(profile)
    validate(cfg)

    evo_runner = EvolutionRunner(cfg.evo, cfg.backend, project_dir, verbose=True)
    evo_runner.run_ray()


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
    typer.echo(f"run_name={run_name} seed={seed} profile={profile}")

    if dry_run:
        config_file = resolve_config_file(project_dir)
        ns = load_config_file(config_file)
        get_config = ns.get("get_config")
        if not callable(get_config):
            raise typer.BadParameter("config.py must define get_config(profile)")
        cfg: RayEvolveConfig = get_config(profile)
        validate(cfg)
        return

    with _ray_session(ray_address, ray_ip, ray_port, ray_debug):
        _run_single_project(project_dir, profile)


@app.command()
def run_batch(
    project_dirs: List[str] = typer.Argument(..., help="One or more project directories"),
    profile: str = typer.Option("default"),
    max_concurrent: int = typer.Option(
        1,
        help="Max projects running simultaneously. 1 = sequential.",
    ),
    ray_debug: bool = typer.Option(False, help="Enable Ray debug env vars"),
    ray_address: str | None = typer.Option(
        None,
        help="Optional Ray Client address (ray://<host>:<port>).",
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
    Run rayevolve on multiple projects. Projects run sequentially by default;
    use --max-concurrent to run multiple projects in parallel.
    """
    # Validate all projects upfront before starting Ray.
    for proj in project_dirs:
        resolve_config_file(proj)

    typer.echo(f"Batch: {len(project_dirs)} projects, max_concurrent={max_concurrent}")

    results: dict[str, tuple[bool, str | None]] = {}

    with _ray_session(ray_address, ray_ip, ray_port, ray_debug):
        if max_concurrent <= 1:
            for proj in project_dirs:
                typer.echo(f"\n{'='*60}\nRunning: {proj}\n{'='*60}")
                try:
                    _run_single_project(proj, profile)
                    results[proj] = (True, None)
                except Exception as e:
                    typer.echo(f"FAILED: {proj} -- {e}", err=True)
                    results[proj] = (False, str(e))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
                futures = {
                    pool.submit(_run_single_project, proj, profile): proj
                    for proj in project_dirs
                }
                for future in concurrent.futures.as_completed(futures):
                    proj = futures[future]
                    try:
                        future.result()
                        results[proj] = (True, None)
                        typer.echo(f"DONE: {proj}")
                    except Exception as e:
                        typer.echo(f"FAILED: {proj} -- {e}", err=True)
                        results[proj] = (False, str(e))

    # Print summary
    typer.echo(f"\n{'='*60}\nBATCH SUMMARY\n{'='*60}")
    for proj, (ok, err) in results.items():
        status = "OK" if ok else f"FAILED: {err}"
        typer.echo(f"  {proj}: {status}")

    failed = sum(1 for ok, _ in results.values() if not ok)
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()