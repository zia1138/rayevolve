import io
import os
import zipfile
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Zip Utilities
# ==============================================================================

def zip_dir_to_bytes(dir_path: str | Path) -> bytes:
    """Compresses a directory into an in-memory zip archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(dir_path):
            # Skip hidden directories in-place so os.walk won't descend into them
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.startswith('.'):
                    continue
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

def parse_results_from_zip(zip_bytes: bytes) -> dict:
    """Parses returncode, results.json, and logs directly from an in-memory zip archive.

    Returns a flat dict with keys: returncode, correct, error, combined_score,
    any other metrics from results.json, stdout_log, stderr_log.
    """
    loaded_results: dict = {"returncode": None, "correct": False, "error": None}
    buffer = io.BytesIO(zip_bytes)

    with zipfile.ZipFile(buffer, 'r') as zf:
        namelist = zf.namelist()

        if "returncode.json" in namelist:
            try:
                loaded_results["returncode"] = json.loads(zf.read("returncode.json").decode("utf-8")).get("returncode")
            except json.JSONDecodeError:
                logger.warning("Could not decode JSON from returncode.json in zip.")

        if "results.json" in namelist:
            try:
                loaded_results.update(json.loads(zf.read("results.json").decode("utf-8")))
            except json.JSONDecodeError:
                logger.warning("Could not decode JSON from results.json in zip.")

        loaded_results["stdout_log"] = zf.read("job_log.out").decode("utf-8") if "job_log.out" in namelist else ""
        loaded_results["stderr_log"] = zf.read("job_log.err").decode("utf-8") if "job_log.err" in namelist else ""

    return loaded_results

class ExecutionBackend(ABC):
    """Abstract base class for execution backends (Ray, Modal, Local, etc.)."""
    @abstractmethod
    def run_job(
        self,
        parent_zip_bytes: bytes,
        generated_code: str,
        exec_fname_rel: str
    ) -> Tuple[Dict[str, Any], float, bytes]:
        """
        Executes the generated code and returns a tuple of (results_dict, runtime_seconds, zip bytes from execution results).
        """
        pass

    @abstractmethod
    def run_command(
        self,
        parent_zip_bytes: bytes,
        cmd: List[str],
    ) -> Dict[str, Any]:
        """
        Runs a command in the parent zip environment and returns a dict with returncode, stdout, and stderr.
        """
        pass

    @abstractmethod
    def run_command_with_zip(
        self,
        parent_zip_bytes: bytes,
        cmd: List[str],
    ) -> Tuple[Dict[str, Any], bytes]:
        """
        Runs a command that may modify files in the parent zip environment.
        Returns (results_dict with returncode/stdout/stderr, updated_zip_bytes).
        """
        pass
