import subprocess
import time
import os
from pathlib import Path
from typing import Optional
from rayevolve.utils import load_results, parse_time_to_seconds
import logging

logger = logging.getLogger(__name__)


class ProcessWithLogging:
    """
    Synchronous wrapper that stores stdout/stderr and return code.

    The process has already completed when this object is created.
    stdout/stderr are available as strings and also written to disk.
    """

    def __init__(
        self,
        pid: int,
        returncode: int,
        stdout_text: str,
        stderr_text: str,
        stdout_path: Path,
        stderr_path: Path,
    ):
        self._pid = pid
        self._returncode = returncode
        self.stdout = stdout_text
        self.stderr = stderr_text
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path

    # Backwards-compatible attributes
    @property
    def pid(self) -> int:
        return self._pid

    @property
    def returncode(self) -> int:
        return self._returncode

    def poll(self) -> int:
        """Return the return code immediately (process is already done)."""
        return self._returncode

    def kill(self):
        """No-op: process already finished (kept for interface compatibility)."""
        logger.debug("kill() called on a completed synchronous process; no action taken.")

    def cleanup_logging(self):
        """No-op kept for interface compatibility."""
        pass

    def __str__(self):
        return f"ProcessWithLogging(PID: {self._pid})"

    def __repr__(self):
        return f"ProcessWithLogging(PID: {self._pid}, returncode: {self._returncode})"
    
    def wait(self):
        pass


def submit(log_dir: str, cmd: list[str], verbose: bool = True):
    """
    Submits a command for local execution (synchronously) and logs output.

    Args:
        log_dir: The directory to store logs.
        cmd: The command and its arguments as a list of strings.
        verbose: Whether to enable verbose logging.

    Returns:
        ProcessWithLogging: Wrapper containing completed process info & logs.
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    stdout_path = log_dir_path / "job_log.out"
    stderr_path = log_dir_path / "job_log.err"

    # Environment tweaks for consistent output behavior
    env = os.environ.copy    ()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    if verbose:
        logger.info(f"Launching local command: {' '.join(cmd)}")

    # Run the process synchronously and capture stdout/stderr
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=10*60, # 10 minutes timeout    
        )
    except subprocess.TimeoutExpired as e:
        print("Timeout command: " + ' '.join(cmd))
        completed = subprocess.CompletedProcess(
            args=cmd,
            returncode=255,
            stdout="Process timed out after 10 minutes.",
            stderr=""
        )

    # Persist outputs to disk
    stdout_text = completed.stdout or ""
    stderr_text = completed.stderr or ""
    try:
        stdout_path.write_text(stdout_text, encoding="utf-8")
    except Exception as e:
        logger.error(f"Error writing stdout to file: {e}")
    try:
        stderr_path.write_text(stderr_text, encoding="utf-8")
    except Exception as e:
        logger.error(f"Error writing stderr to file: {e}")

    if verbose:
        logger.info(f"Completed local command with return code: {completed.returncode}")

    # Build a synchronous wrapper; set pid to 0 as requested
    wrapped_process = ProcessWithLogging(
        pid=0,
        returncode=completed.returncode,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    return wrapped_process


def monitor(
    process: ProcessWithLogging,
    results_dir: str,
    poll_interval: int = 10,
    verbose: bool = False,
    timeout: Optional[str] = None,
):
    """
    Monitors a (synchronous) local process and loads its results.

    Since submit() is synchronous and the process is done by the time this is called,
    this returns immediately with the same output as load_results(results_dir).
    """
    if verbose:
        logger.info(f"Monitoring local process with PID: {process.pid} (completed).")
        logger.info(f"Process completed with return code: {process.returncode}")

    # No polling; just return results immediately
    return load_results(results_dir)