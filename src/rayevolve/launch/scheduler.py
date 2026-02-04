import logging
import time
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, Tuple, Union, List
from .local_sync import submit as submit_local, monitor as monitor_local
from .local_sync import ProcessWithLogging
from rayevolve.utils import parse_time_to_seconds

from rayevolve.core.common import JobConfig

logger = logging.getLogger(__name__)


class JobScheduler:
    def __init__(
        self,
        job_type: str,
        config: JobConfig,
        project_dir: str,
        verbose: bool = True,
    ):
        self.job_type = job_type
        self.config = config
        self.project_dir = project_dir
        self.verbose = verbose

        if self.job_type == "local":
            self.monitor = monitor_local
        else:
            raise ValueError(
                f"Unknown job type: {job_type}. "
                f"Must be 'local'"
            )

    def _build_command(self, exec_fname_t: str, results_dir_t: str) -> List[str]:
        # For local jobs, check if conda environment is specified
        if (
            self.job_type == "local"
            and isinstance(self.config, JobConfig)
            and self.config.conda_env
        ):
            # Use conda run to execute in specific environment
            cmd = [
                "conda",
                "run",
                "-n",
                self.config.conda_env,
                "python",
                f"{self.config.eval_program_path}",
                "--program_path",
                f"{exec_fname_t}",
                "--results_dir",
                results_dir_t,
            ]
        else:
            cmd = [
                "python",
                f"{self.config.eval_program_path}",
                "--program_path",
                f"{exec_fname_t}",
                "--results_dir",
                results_dir_t,
            ]
        if self.config.extra_cmd_args:
            for k, v in self.config.extra_cmd_args.items():
                cmd.extend([f"--{k}", str(v)])
        return cmd

    def run(
        self, exec_fname_t: str, results_dir_t: str
    ) -> Tuple[Dict[str, Any], float]:
        job_id: Union[str, ProcessWithLogging]
        cmd = self._build_command(exec_fname_t, results_dir_t)
        start_time = time.time()

        if self.job_type == "local":
            assert isinstance(self.config, JobConfig)
            job_id = submit_local(results_dir_t, cmd, self.project_dir, verbose=self.verbose)
        else:
            raise ValueError(f"Unknown job type: {self.job_type}")

        results = monitor_local(job_id, results_dir_t)

        end_time = time.time()
        rtime = end_time - start_time

        # Ensure results is not None
        if results is None:
            results = {"correct": {"correct": False}, "metrics": {}}

        return results, rtime

    def submit_async(
        self, exec_fname_t: str, results_dir_t: str
    ) -> Union[str, ProcessWithLogging]:
        """Submit a job asynchronously and return the job ID or process."""
        cmd = self._build_command(exec_fname_t, results_dir_t)
        if self.job_type == "local":
            assert isinstance(self.config, JobConfig)
            return submit_local(results_dir_t, cmd, self.project_dir, verbose=self.verbose)
        raise ValueError(f"Unknown job type: {self.job_type}")

    def get_job_results(
        self, job_id: Union[str, ProcessWithLogging], results_dir: str
    ) -> Optional[Dict[str, Any]]:
        """Get results from a completed job."""
        job_id.wait()
        return monitor_local(
            job_id,
            results_dir,
            verbose=self.verbose,
            timeout=self.config.time,
        )
