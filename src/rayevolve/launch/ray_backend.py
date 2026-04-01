import os
import tempfile
import subprocess
import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List

import ray

from rayevolve.core.common import BackendConfig
from rayevolve.launch.common import (
    ExecutionBackend,
    zip_dir_to_bytes,
    extract_bytes_to_dir,
    parse_results_from_zip
)

logger = logging.getLogger(__name__)

# ==============================================================================
# 2. Remote Ray Execution Task
# ==============================================================================

@ray.remote
def ray_evaluator_task(
    parent_zip_bytes: bytes,
    generated_code: str,
    exec_fname_rel: str,
    cmd: List[str],
    timeout_sec: int
) -> bytes:
    """
    Runs an evaluation job in an isolated temporary directory on a Ray worker node.
    Returns the zipped result directory (includes returncode.json, job_log.out, job_log.err).
    """
    import json as _json

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Extract the parent program files
        extract_bytes_to_dir(parent_zip_bytes, temp_dir)

        # 2. Write the generated code to the specified relative path (if provided)
        if generated_code and exec_fname_rel:
            target_file = temp_path / exec_fname_rel
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(generated_code, encoding="utf-8")

        # 3. Setup environment variables
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env.pop("VIRTUAL_ENV", None) # TODO: Make this a parameter for only when uv is used.

        # 4. Execute the command
        try:
            cp = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=temp_dir,
                timeout=timeout_sec,
            )
            stdout_text = cp.stdout or ""
            stderr_text = cp.stderr or ""
            returncode = cp.returncode
        except subprocess.TimeoutExpired:
            stdout_text = f"Process timed out after {timeout_sec} seconds."
            stderr_text = ""
            returncode = 255

        # 5. Save logs and returncode so they are included in the returned zip
        (temp_path / "job_log.out").write_text(stdout_text, encoding="utf-8")
        (temp_path / "job_log.err").write_text(stderr_text, encoding="utf-8")
        (temp_path / "returncode.json").write_text(_json.dumps({"returncode": returncode}), encoding="utf-8")

        # 6. Zip the entire temp directory and return
        return zip_dir_to_bytes(temp_dir)

@ray.remote
def ray_run_command_task(
    parent_zip_bytes: bytes,
    cmd: List[str],
    timeout_sec: int = 60
) -> Dict[str, Any]:
    """
    Extracts parent_zip_bytes to a temp directory, runs a command, and returns
    a dict with returncode, stdout, and stderr.
    """
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        extract_bytes_to_dir(parent_zip_bytes, temp_dir)

        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env.pop("VIRTUAL_ENV", None) # TODO: Make this a parameter for only when uv is used.

        try:
            cp = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=temp_dir,
                timeout=timeout_sec,
            )
            return {
                "returncode": cp.returncode,
                "stdout": cp.stdout or "",
                "stderr": cp.stderr or "",
            }
        except subprocess.TimeoutExpired:
            return {
                "returncode": 255,
                "stdout": "",
                "stderr": f"Command timed out after {timeout_sec} seconds.",
            }

class RayExecutionBackend(ExecutionBackend):
    """
    Ray-based implementation of the execution backend. 
    It ships the project to remote nodes via in-memory zip files.
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
            logger.info(f"Submitting remote job for {exec_fname_rel}...")

        # Ship the job to the Ray cluster
        future = ray_evaluator_task.remote(
            parent_zip_bytes=parent_zip_bytes,
            generated_code=generated_code,
            exec_fname_rel=exec_fname_rel,
            cmd=cmd,
            timeout_sec=self.config.timeout_sec
        )
        
        result_zip_bytes: bytes = ray.get(future)
        rtime = time.time() - t0

        # Parse the results directly from the zip bytes
        results = parse_results_from_zip(result_zip_bytes)
        returncode = results.get("returncode")

        if self.verbose:
            logger.info(f"Remote job completed in {rtime:.2f}s with return code: {returncode}")

        # Communicate this failure back to LLM by appending to stderr_log.
        if returncode is not None and returncode != 0:
            results["stderr_log"] += f"\nProcess failed with return code {returncode}."

        return results, rtime, result_zip_bytes

    def run_command(
        self,
        parent_zip_bytes: bytes,
        cmd: List[str],
    ) -> Dict[str, Any]:
        future = ray_run_command_task.remote(
            parent_zip_bytes=parent_zip_bytes,
            cmd=cmd,
            timeout_sec=self.config.timeout_sec,
        )
        return ray.get(future)

    def run_command_with_zip(
        self,
        parent_zip_bytes: bytes,
        cmd: List[str],
    ) -> Tuple[Dict[str, Any], bytes]:
        future = ray_evaluator_task.remote(
            parent_zip_bytes=parent_zip_bytes,
            generated_code="",
            exec_fname_rel="",
            cmd=cmd,
            timeout_sec=self.config.timeout_sec,
        )
        result_zip_bytes: bytes = ray.get(future)
        results = parse_results_from_zip(result_zip_bytes)
        return results, result_zip_bytes
