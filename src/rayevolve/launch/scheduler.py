import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rayevolve.core.common import JobConfig
from rayevolve.utils import load_results

logger = logging.getLogger(__name__)


def run_local(
    log_dir: str,
    cmd: List[str],
    cwd: str,
    *,
    verbose: bool = True,
    timeout_sec: int,
) -> int:
    """Run a command synchronously, log stdout/stderr, and return returncode."""
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
    def __init__(self, config: JobConfig, project_dir: str, verbose: bool = True):
        self.config = config
        self.project_dir = project_dir
        self.verbose = verbose

    def _build_command(self, exec_fname_t: str, results_dir_t: str) -> List[str]:
        base = (
            ["conda", "run", "-n", self.config.conda_env]
            if self.config.conda_env
            else []
        )

        cmd = [
            *base,
            "python",
            "evaluate.py",
            "--program_path",
            exec_fname_t,
            "--results_dir",
            results_dir_t,
        ]

        if self.config.extra_cmd_args:
            for k, v in self.config.extra_cmd_args.items():
                flag = f"--{k}" if not str(k).startswith("--") else str(k)
                cmd.extend([flag, str(v)])

        return cmd

    def run(self, exec_fname_t: str, results_dir_t: str) -> Tuple[Dict[str, Any], float]:
        cmd = self._build_command(exec_fname_t, results_dir_t)

        t0 = time.time()
        _returncode = run_local(
            results_dir_t, cmd, self.project_dir, verbose=self.verbose, timeout_sec=self.config.timeout_sec
        )

        results = load_results(results_dir_t) or {
            "correct": {"correct": False},
            "metrics": {},
        }

        rtime = time.time() - t0
        return results, rtime