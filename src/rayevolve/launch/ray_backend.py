import io
import os
import zipfile
import tempfile
import subprocess
import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List
from abc import ABC, abstractmethod

import ray

from rayevolve.core.common import JobConfig

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Zip Utilities
# ==============================================================================

def zip_dir_to_bytes(dir_path: str | Path) -> bytes:
    """Compresses a directory into an in-memory zip archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Keep paths relative to the target directory root
                arcname = os.path.relpath(file_path, dir_path)
                zf.write(file_path, arcname)
    return buffer.getvalue()


def extract_bytes_to_dir(zip_bytes: bytes, extract_to: str | Path) -> None:
    """Extracts an in-memory zip archive to a target directory."""
    buffer = io.BytesIO(zip_bytes)
    with zipfile.ZipFile(buffer, 'r') as zf:
        zf.extractall(extract_to)

# ==============================================================================
# 2. Remote Ray Execution Task
# ==============================================================================

@ray.remote
def ray_evaluator_task(
    project_zip_bytes: bytes,
    generated_code: str,
    exec_fname_rel: str,
    cmd: List[str],
    timeout_sec: int
) -> Tuple[int, bytes]:
    """
    Runs an evaluation job in an isolated temporary directory on a Ray worker node.
    """
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. Extract the original project files
        extract_bytes_to_dir(project_zip_bytes, temp_dir)
        
        # 2. Write the generated code to the specified relative path
        target_file = temp_path / exec_fname_rel
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(generated_code, encoding="utf-8")
        
        # 3. Setup environment variables
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        
        stdout_path = temp_path / "job_log.out"
        stderr_path = temp_path / "job_log.err"
        
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
            
        # 5. Save logs so they are included in the returned zip
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        
        # 6. Zip the entire temp directory and return
        result_zip_bytes = zip_dir_to_bytes(temp_dir)
        
        return returncode, result_zip_bytes

# ==============================================================================
# 3. Backend Interfaces & Implementation
# ==============================================================================

import json

def parse_results_from_zip(zip_bytes: bytes) -> dict:
    """Parses metrics, correct, and logs directly from an in-memory zip archive."""
    loaded_results = {"correct": {"correct": False}, "metrics": {}}
    buffer = io.BytesIO(zip_bytes)
    
    with zipfile.ZipFile(buffer, 'r') as zf:
        namelist = zf.namelist()
        
        if "metrics.json" in namelist:
            try:
                loaded_results["metrics"] = json.loads(zf.read("metrics.json").decode("utf-8"))
            except json.JSONDecodeError:
                logger.warning("Could not decode JSON from metrics.json in zip.")
                
        if "correct.json" in namelist:
            try:
                loaded_results["correct"] = json.loads(zf.read("correct.json").decode("utf-8"))
            except json.JSONDecodeError:
                pass
                
        if "job_log.out" in namelist:
            loaded_results["stdout_log"] = zf.read("job_log.out").decode("utf-8")
        else:
            loaded_results["stdout_log"] = ""
            
        if "job_log.err" in namelist:
            loaded_results["stderr_log"] = zf.read("job_log.err").decode("utf-8")
        else:
            loaded_results["stderr_log"] = ""
            
    return loaded_results

class ExecutionBackend(ABC):
    """Abstract base class for execution backends (Ray, Modal, Local, etc.)."""
    @abstractmethod
    def run_job(
        self, 
        generated_code: str, 
        exec_fname_rel: str
    ) -> Tuple[Dict[str, Any], float]:
        """
        Executes the generated code and returns a tuple of (results_dict, runtime_seconds).
        """
        pass


class RayExecutionBackend(ExecutionBackend):
    """
    Ray-based implementation of the execution backend. 
    It ships the project to remote nodes via in-memory zip files.
    """
    def __init__(self, config: JobConfig, project_dir: str = None, project_zip_bytes: bytes = None, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        
        if project_zip_bytes is not None:
            self.project_zip_bytes = project_zip_bytes
            self.project_dir = None
        elif project_dir is not None:
            self.project_dir = Path(project_dir).resolve()
            # Pre-zip the project directory once during initialization.
            # This prevents re-zipping the same files for every evaluation.
            if self.verbose:
                logger.info(f"Pre-zipping project directory: {self.project_dir}")
            self.project_zip_bytes = zip_dir_to_bytes(self.project_dir)
        else:
            raise ValueError("Either project_dir or project_zip_bytes must be provided.")

    def _build_command(self) -> List[str]:
        base = (
            ["conda", "run", "-n", self.config.conda_env]
            if self.config.conda_env
            else []
        )
        cmd = [
            *base,
            "python",
            "evaluate.py",
        ]

        return cmd

    def run_job(
        self, 
        generated_code: str, 
        exec_fname_rel: str
    ) -> Tuple[Dict[str, Any], float]:
        
        # We tell evaluate.py to output results to ".", which is the root of the temp dir
        cmd = self._build_command()
        
        t0 = time.time()
        
        if self.verbose:
            logger.info(f"Submitting remote job for {exec_fname_rel}...")
            
        # Ship the job to the Ray cluster
        future = ray_evaluator_task.remote(
            project_zip_bytes=self.project_zip_bytes,
            generated_code=generated_code,
            exec_fname_rel=exec_fname_rel,
            cmd=cmd,
            timeout_sec=self.config.timeout_sec
        )
        
        returncode, result_zip_bytes = ray.get(future)
        rtime = time.time() - t0
        
        if self.verbose:
            logger.info(f"Remote job completed in {rtime:.2f}s with return code: {returncode}")
            
        # Parse the results directly from the zip bytes
        results = parse_results_from_zip(result_zip_bytes)
        
        # Add a default error message if the process failed but no explicit error was provided

        # TODO: Need to communicate this info back to LLM agents so than can respond.
        if returncode == 255 and "timeout" in results.get("stdout_log", "").lower():
            results["error"] = f"Execution timed out after {self.config.timeout_sec} seconds."
        elif returncode != 0 and not results.get("error"):
            results["error"] = f"Process failed with return code {returncode}. See stderr_log."
        
        return results, rtime
