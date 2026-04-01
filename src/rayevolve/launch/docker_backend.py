import os
import tempfile
import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List
import json as _json

import ray
import docker
import requests
from docker.errors import ImageNotFound

from rayevolve.core.common import BackendConfig
from rayevolve.launch.common import (
    ExecutionBackend,
    zip_dir_to_bytes,
    extract_bytes_to_dir,
    parse_results_from_zip
)

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Docker Utilities
# ==============================================================================

def _get_user_ids() -> str:
    """Returns UID:GID for the current user to ensure file permission parity."""
    try:
        return f"{os.getuid()}:{os.getgid()}"
    except AttributeError:
        # Fallback for non-POSIX systems
        return ""

def _get_docker_config(package_manager: str) -> Tuple[str, str, Dict[str, str]]:
    """Returns the image name, cache volume bind string, and env vars based on the package manager."""
    if package_manager == "uv":
        return (
            "ghcr.io/astral-sh/uv:python3.11-bookworm-slim",
            "/root/.cache/uv",
            {"UV_PROJECT_ENVIRONMENT": "/tmp/.venv"}
        )
    elif package_manager == "pixi":
        return (
            "ghcr.io/prefix-dev/pixi:latest",
            "/root/.cache/pixi",
            {}
        )
    else:
        # Fallback
        return (
            "python:3.11-slim",
            "/root/.cache/pip",
            {}
        )

def run_in_docker(
    client: docker.DockerClient,
    temp_dir: str,
    cmd: List[str],
    timeout_sec: int,
    package_manager: str
) -> Tuple[int, str, str]:
    """Runs a command inside a Docker container using the Python SDK and returns (returncode, stdout, stderr)."""
    image, cache_mount_path, extra_env = _get_docker_config(package_manager)
    
    # Ensure image exists
    try:
        client.images.get(image)
    except ImageNotFound:
        client.images.pull(image)

    volumes = {
        os.path.abspath(temp_dir): {"bind": "/workspace", "mode": "rw"},
        f"ray_{package_manager}_cache": {"bind": cache_mount_path, "mode": "rw"}
    }
    
    env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        **extra_env
    }

    container = client.containers.run(
        image=image,
        command=cmd,
        volumes=volumes,
        working_dir="/workspace",
        user=_get_user_ids(),
        environment=env,
        detach=True
    )

    try:
        result = container.wait(timeout=timeout_sec)
        returncode = result["StatusCode"]
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
        returncode = 255
        try:
            container.stop(timeout=1)
        except Exception:
            pass
    finally:
        stdout_bytes, stderr_bytes = container.logs(demux=True)
        stdout_text = stdout_bytes.decode("utf-8") if stdout_bytes else ""
        stderr_text = stderr_bytes.decode("utf-8") if stderr_bytes else ""
        
        if returncode == 255:
            stderr_text += f"\n[Execution timed out after {timeout_sec}s]"
        
        container.remove(force=True)

    return returncode, stdout_text, stderr_text

# ==============================================================================
# 2. Remote Ray Execution Tasks
# ==============================================================================

@ray.remote
def ray_evaluator_task_docker(
    parent_zip_bytes: bytes,
    generated_code: str,
    exec_fname_rel: str,
    cmd: List[str],
    timeout_sec: int,
    package_manager: str
) -> bytes:
    """
    Runs an evaluation job in an isolated Docker container on a Ray worker node.
    Returns the zipped result directory (includes returncode.json, job_log.out, job_log.err).
    """
    client = docker.from_env()

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Extract the parent program files
        extract_bytes_to_dir(parent_zip_bytes, temp_dir)

        # 2. Write the generated code to the specified relative path (if provided)
        if generated_code and exec_fname_rel:
            target_file = temp_path / exec_fname_rel
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(generated_code, encoding="utf-8")

        # 3. Execute the command in Docker
        returncode, stdout_text, stderr_text = run_in_docker(
            client=client,
            temp_dir=temp_dir,
            cmd=cmd,
            timeout_sec=timeout_sec,
            package_manager=package_manager
        )

        # 4. Save logs and returncode so they are included in the returned zip
        (temp_path / "job_log.out").write_text(stdout_text, encoding="utf-8")
        (temp_path / "job_log.err").write_text(stderr_text, encoding="utf-8")
        (temp_path / "returncode.json").write_text(_json.dumps({"returncode": returncode}), encoding="utf-8")

        # 5. Zip the entire temp directory and return
        return zip_dir_to_bytes(temp_dir)

@ray.remote
def ray_run_command_task_docker(
    parent_zip_bytes: bytes,
    cmd: List[str],
    timeout_sec: int,
    package_manager: str
) -> Dict[str, Any]:
    """
    Extracts parent_zip_bytes to a temp directory, runs a command in Docker, and returns
    a dict with returncode, stdout, and stderr.
    """
    client = docker.from_env()

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        extract_bytes_to_dir(parent_zip_bytes, temp_dir)

        returncode, stdout_text, stderr_text = run_in_docker(
            client=client,
            temp_dir=temp_dir,
            cmd=cmd,
            timeout_sec=timeout_sec,
            package_manager=package_manager
        )

        return {
            "returncode": returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
        }

# ==============================================================================
# 3. Docker Backend Implementation
# ==============================================================================

class DockerExecutionBackend(ExecutionBackend):
    """
    Ray-based implementation of the execution backend that uses Docker.
    It ships the project to remote nodes via in-memory zip files, and executes
    them inside isolated Docker containers using the Docker SDK.
    """
    def __init__(self, config: BackendConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose

    def _build_command(self) -> List[str]:
        base = []
        if self.config.package_manager == "uv":
            base = ["uv", "-q", "run"]
        elif self.config.package_manager == "pixi":
            base = ["pixi", "run"]
        cmd = [
            *base,
            "python",
            "evaluate.py",
        ]

        return cmd

    def run_job(
        self,
        parent_zip_bytes: bytes,
        generated_code: str,
        exec_fname_rel: str
    ) -> Tuple[Dict[str, Any], float, bytes]:

        cmd = self._build_command()

        t0 = time.time()

        if self.verbose:
            logger.info(f"Submitting remote Docker job for {exec_fname_rel}...")

        # Ship the job to the Ray cluster
        future = ray_evaluator_task_docker.remote(
            parent_zip_bytes=parent_zip_bytes,
            generated_code=generated_code,
            exec_fname_rel=exec_fname_rel,
            cmd=cmd,
            timeout_sec=self.config.timeout_sec,
            package_manager=self.config.package_manager
        )
        
        result_zip_bytes: bytes = ray.get(future)
        rtime = time.time() - t0

        # Parse the results directly from the zip bytes
        results = parse_results_from_zip(result_zip_bytes)
        returncode = results.get("returncode")

        if self.verbose:
            logger.info(f"Remote Docker job completed in {rtime:.2f}s with return code: {returncode}")

        # Communicate this failure back to LLM by appending to stderr_log.
        if returncode is not None and returncode != 0:
            results["stderr_log"] += f"\nProcess failed with return code {returncode}."

        return results, rtime, result_zip_bytes

    def run_command(
        self,
        parent_zip_bytes: bytes,
        cmd: List[str],
    ) -> Dict[str, Any]:
        future = ray_run_command_task_docker.remote(
            parent_zip_bytes=parent_zip_bytes,
            cmd=cmd,
            timeout_sec=self.config.timeout_sec,
            package_manager=self.config.package_manager
        )
        return ray.get(future)

    def run_command_with_zip(
        self,
        parent_zip_bytes: bytes,
        cmd: List[str],
    ) -> Tuple[Dict[str, Any], bytes]:
        future = ray_evaluator_task_docker.remote(
            parent_zip_bytes=parent_zip_bytes,
            generated_code="",
            exec_fname_rel="",
            cmd=cmd,
            timeout_sec=self.config.timeout_sec,
            package_manager=self.config.package_manager
        )
        result_zip_bytes: bytes = ray.get(future)
        results = parse_results_from_zip(result_zip_bytes)
        return results, result_zip_bytes
