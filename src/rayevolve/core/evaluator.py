from pathlib import Path
import importlib.util
import sys
import uuid
import subprocess
import shlex
import re

import json
from typing import Any, Dict, Optional, List, Tuple, Union

# --- Existing Helpers ---

def load_module_from_path(file_path: str | Path, unique: bool = True):
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    module_name = path.stem
    if unique:
        module_name = f"{module_name}_{uuid.uuid4().hex}"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def save_json_results(
    results_dir: str | Path,
    metrics: Dict[str, Any],
    correct: bool,
    error: Optional[str] = None,
) -> None:
    """Saves metrics and correctness status to JSON files."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    correct_payload = {"correct": correct, "error": error}
    correct_file = results_path / "correct.json"
    with open(correct_file, "w") as f:
        json.dump(correct_payload, f, indent=4)
    #print(f"Correctness and error status saved to {correct_file}")

    metrics_file = results_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    #print(f"Metrics saved to {metrics_file}")    

# --- New Syntactic Sugar Helpers ---

def run_command(
    cmd: List[str] | str,
    cwd: Optional[str | Path] = None,
    timeout: float = 30.0,
    check: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """
    Run a shell command with a timeout and return the result.
    
    Args:
        cmd: Command to run (list of strings or string).
        cwd: Working directory.
        timeout: Timeout in seconds.
        check: If True, raise CalledProcessError on non-zero exit code.
        env: Environment variables.

    Returns:
        CompletedProcess object with stdout/stderr.
    """
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=check,
            env=env
        )
        return result
    except subprocess.TimeoutExpired as e:
        # Re-raise with captured output if available
        raise subprocess.TimeoutExpired(
            e.cmd, e.timeout, output=e.stdout, stderr=e.stderr
        ) from e

def compile_and_run(
    source_path: str | Path,
    language: str = "auto",
    args: List[str] = [],
    cwd: Optional[str | Path] = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess:
    """
    Compile (if necessary) and run a program.
    Supports: python, cpp, c, rust, go.
    
    Args:
        source_path: Path to the source file.
        language: Language of the source file ('auto' detects from extension).
        args: Additional arguments to pass to the compiled executable/script.
        cwd: Working directory.
        timeout: Timeout for the run step.

    Returns:
        The result of the execution.
    """
    path = Path(source_path).resolve()
    if cwd is None:
        cwd = path.parent
    
    if language == "auto":
        ext = path.suffix.lower()
        if ext == ".py": language = "python"
        elif ext == ".cpp": language = "cpp"
        elif ext == ".c": language = "c"
        elif ext == ".rs": language = "rust"
        elif ext == ".go": language = "go"
        else: raise ValueError(f"Unknown language for extension {ext}")

    # Build Step
    exe_path = path.with_suffix("") # standard executable name (no ext on linux/mac)
    # Windows might need .exe, but for now assuming linux/mac/docker env
    
    if language == "cpp":
        run_command(["g++", "-O3", str(path), "-o", str(exe_path)], cwd=cwd, check=True)
        cmd = [str(exe_path)] + args
    elif language == "c":
        run_command(["gcc", "-O3", str(path), "-o", str(exe_path)], cwd=cwd, check=True)
        cmd = [str(exe_path)] + args
    elif language == "rust":
        run_command(["rustc", "-O", str(path), "-o", str(exe_path)], cwd=cwd, check=True)
        cmd = [str(exe_path)] + args
    elif language == "go":
        run_command(["go", "build", "-o", str(exe_path), str(path)], cwd=cwd, check=True)
        cmd = [str(exe_path)] + args
    elif language == "python":
        cmd = [sys.executable, str(path)] + args
    else:
        raise ValueError(f"Unsupported language: {language}")

    # Run Step
    return run_command(cmd, cwd=cwd, timeout=timeout)

def docker_run_command(
    image: str,
    cmd: List[str] | str,
    cwd: str | Path,
    mounts: List[Tuple[str, str]] = [],
    timeout: float = 30.0
) -> subprocess.CompletedProcess:
    """
    Run a command inside a Docker container.
    Automatically mounts the 'cwd' to '/app' and sets it as working directory.
    
    Args:
        image: Docker image name (e.g., 'python:3.10').
        cmd: Command to run inside container.
        cwd: Local directory to mount as /app.
        mounts: Additional (host_path, container_path) tuples to mount.
        timeout: Timeout in seconds.
    """
    cwd = Path(cwd).resolve()
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{cwd}:/app",
        "-w", "/app"
    ]
    
    for host_path, container_path in mounts:
        docker_cmd.extend(["-v", f"{Path(host_path).resolve()}:{container_path}"])
    
    docker_cmd.append(image)
    
    if isinstance(cmd, str):
        docker_cmd.extend(shlex.split(cmd))
    else:
        docker_cmd.extend(cmd)
        
    return run_command(docker_cmd, timeout=timeout)

def parse_json_from_stdout(stdout: str) -> Dict[str, Any]:
    """
    robustly extract JSON from stdout. 
    Handles cases where there is noise before/after the JSON object.
    Finds the first '{' and last '}' to extract the main object.
    """
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        pass
        
    # Try to find a JSON object block
    match = re.search(r'(\{.*\})', stdout, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
            
    raise ValueError("Could not parse valid JSON from stdout")
