import logging
import os
import subprocess
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rayevolve.core.common import BackendConfig

logger = logging.getLogger(__name__)


def load_results(results_dir: str):
    """
    Loads results from the specified directory.
    Reads in job_log.out, job_log.err, metrics.json, and 
    correct.json if they exist.    
    
    Args:
        results_dir: The directory containing the results.

    Returns:
        dict: A dictionary containing the loaded results.
    """
    loaded_results = {"correct": {"correct": False}, "metrics": {}}
    results_dir_path = Path(results_dir)

    stdout_log_path = results_dir_path / "job_log.out"
    if stdout_log_path.exists():
        with open(stdout_log_path, "r") as f:
            loaded_results["stdout_log"] = f.read()
    else:
        loaded_results["stdout_log"] = ""

    stderr_log_path = results_dir_path / "job_log.err"
    if stderr_log_path.exists():
        with open(stderr_log_path, "r") as f:
            loaded_results["stderr_log"] = f.read()
    else:
        loaded_results["stderr_log"] = ""

    metrics_file_path = results_dir_path / "metrics.json"
    if metrics_file_path.exists():
        with open(metrics_file_path, "r") as f:
            try:
                loaded_results["metrics"] = json.load(f)
            except json.JSONDecodeError:
                file_path_str = str(metrics_file_path)
                warning_msg = f"Could not decode JSON from {file_path_str}"
                logger.warning(warning_msg)
                loaded_results["metrics"] = {}
    else:
        file_path_str = str(metrics_file_path)
        warning_msg = f"Metrics file not found at {file_path_str}"
        logger.warning(warning_msg)
        loaded_results["metrics"] = {}

    correct_file_path = results_dir_path / "correct.json"
    if correct_file_path.exists():
        with open(correct_file_path, "r") as f:
            loaded_results["correct"] = json.load(f)
    else:
        loaded_results["correct"] = {"correct": False}

    return loaded_results



def run_local(
    log_dir: str,
    cmd: List[str],
    cwd: str,
    *,
    verbose: bool = True,
    timeout_sec: int,
) -> int:
    """Run a command synchronously, log stdout/stderr to job_log.out and job_log.err, and return returncode."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    stdout_path = log_path / "job_log.out"
    stderr_path = log_path / "job_log.err"

    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")

    if verbose:
        logger.info("Launching local command: %s", " ".join(cmd))

    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
            timeout=timeout_sec,
        )
        stdout_text = cp.stdout or ""
        stderr_text = cp.stderr or ""
        returncode = cp.returncode
    except subprocess.TimeoutExpired:
        stdout_text = f"Process timed out after {timeout_sec} seconds.\n"
        stderr_text = ""
        returncode = 255
        if verbose:
            logger.warning("Timeout running command: %s", " ".join(cmd))

    # Best-effort persist logs
    try:
        stdout_path.write_text(stdout_text, encoding="utf-8")
    except Exception:
        logger.exception("Error writing stdout to %s", stdout_path)

    try:
        stderr_path.write_text(stderr_text, encoding="utf-8")
    except Exception:
        logger.exception("Error writing stderr to %s", stderr_path)

    if verbose:
        logger.info("Completed local command with return code: %s", returncode)

    return returncode


class JobScheduler:
    def __init__(self, config: BackendConfig, project_dir: str, verbose: bool = True):
        self.config = config
        self.project_dir = project_dir
        self.verbose = verbose

    def _build_command(self, exec_fname_t: str, results_dir_t: str) -> List[str]:
        
        # Add conda environment if required.
        base = (
            ["conda", "run", "-n", self.config.conda_env]
            if self.config.conda_env
            else []
        )

        # Runs python evaluate.py --program_path {exec_fname_t} --results_dir {results_dir_t}
        # Assumes user created evaluate.py to use program path and output to results dir.
        cmd = [
            *base,
            "python",
            "evaluate.py",
            "--program-path",
            exec_fname_t,
            "--results-dir",
            results_dir_t,
        ]

        if self.config.extra_cmd_args:
            for k, v in self.config.extra_cmd_args.items():
                flag = f"--{k}" if not str(k).startswith("--") else str(k)
                cmd.extend([flag, str(v)])

        return cmd

    def run(self, exec_fname_t: str, results_dir_t: str) -> Tuple[Dict[str, Any], float]:

        # 1. Build command to run
        cmd = self._build_command(exec_fname_t, results_dir_t)

        # 2. Run command locally and capture return code
        t0 = time.time()
        _returncode = run_local(
            results_dir_t, cmd, self.project_dir, verbose=self.verbose, timeout_sec=self.config.timeout_sec
        )

        # 3. Load job_log.out, job_log.err, metrics.json, and correct.json 
        #    Assumes evaluate.py will ouput metrics.json and correct.json. 
        #    run_local will capture and output job_log.out and job_log.err. 
        # 
        #    Minimally metrics should have a "combined_score" field which is used for evolution.
        results = load_results(results_dir_t) or {
            "correct": {"correct": False},
            "metrics": {},
        }
        
        # 4. Return results and runtime in seconds. 
        rtime = time.time() - t0
        return results, rtime