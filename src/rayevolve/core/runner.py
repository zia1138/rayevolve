import time
import uuid
import logging
from rich.logging import RichHandler
from typing import List, Optional, Union, cast
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
import ray
from rayevolve.database.dbase import ProgramDatabase, Program
from .worker2 import EvoWorker, EvoGen
from .common import EvolutionConfig, DatabaseConfig, JobConfig, FOLDER_PREFIX
from rayevolve.launch.ray_backend import RayExecutionBackend

# Set up logging
logger = logging.getLogger(__name__)

## NOTE: EvolutionRunner still runs in the ray driver. 
class EvolutionRunner:
    def __init__(
        self,
        evo_config: EvolutionConfig,
        job_config: JobConfig,
        db_config: DatabaseConfig,
        project_dir: str, 
        verbose: bool = False,
    ):
        self.evo_config = evo_config
        self.job_config = job_config
        self.db_config = db_config
        self.project_dir = project_dir
        self.verbose = verbose

        # Get full path of results directory.
        if evo_config.results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = Path(f"results_{timestamp}").resolve()
        else:
            self.results_dir = Path(evo_config.results_dir).resolve()

        if self.verbose:
            # Create log file path in results directory
            log_filename = f"{self.results_dir}/evolution_run.log"
            Path(self.results_dir).mkdir(parents=True, exist_ok=True)

            # Set up logging with both console and file handlers
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                handlers=[
                    RichHandler(
                        show_time=False, show_level=False, show_path=False
                    ),  # Console output (clean)
                    logging.FileHandler(
                        log_filename, mode="a", encoding="utf-8"
                    ),  # File output (detailed)
                ],
            )

            # Also log the initial setup information
            logger.info("=" * 80)
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Evolution run started at {start_time}")
            logger.info(f"Results directory: {self.results_dir}")
            logger.info(f"Log file: {log_filename}")
            logger.info("=" * 80)

        # Check if we are resuming a run
        self.resuming_run = False
        self.start_gen = 0
        db_path = Path(f"{self.results_dir}/evolution_db.sqlite")
        if self.evo_config.results_dir is not None and db_path.exists():
            self.resuming_run = True

        self.db = ProgramDatabase.remote(
            db_path_str=str(db_path),
            config=db_config
        )

        self.backend = RayExecutionBackend(
            config=job_config,
            project_dir=self.project_dir,
            verbose=verbose,
        )
        
        # TODO: Need to handle extension of output files since trying to make
        # code language agnostic.
        self.lang_ext = "py"

        if self.resuming_run:
            completed_generations:int = ray.get(self.db.get_last_iteration.remote()) 
            logger.info("=" * 80)
            logger.info("RESUMING PREVIOUS EVOLUTION RUN")
            logger.info("=" * 80)
            logger.info(
                f"Resuming evolution from: {self.results_dir}\n"
                f"Found {completed_generations} "
                "previously completed generations."
            )
            self.start_gen = completed_generations
            logger.info("=" * 80)
            raise NotImplementedError("Resuming runs is not currently supported. This will be implemented in a future update.")


    def run_ray(self):
        """Ray based evolution."""

        if not self.resuming_run:
            self._run_generation_0()

        gen = EvoGen.remote(self.start_gen)  # generation counter

        all_refs = []
        for worker_id in range(self.evo_config.num_agent_workers):
            worker = EvoWorker.remote(
                str(worker_id),
                gen,
                self.evo_config,
                self.job_config,
                self.backend.project_zip_bytes,
                self.results_dir,
                self.db,
                self.verbose,
            )
            all_refs.append(worker.run.remote())

        # TODO: Need to seralize the database here occasionally so restart is possible.
        # TODO: Collect state from the workers and the database.

        # Now wait for ALL workers to finish
        ray.get(all_refs)


    def _run_generation_0(self):
        """Setup and run generation 0 to initialize the database."""
        if self.verbose:
            logger.info(
                f"Reading initial program from {self.project_dir}/main.py"
            )
        
        try:
            initial_code = Path(f"{self.project_dir}/main.py").read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Could not read initial program from {self.project_dir}/main.py. Error: {e}")

        # Run the evaluation code using the Ray backend.
        results, rtime = self.backend.run_job(
            generated_code=initial_code,
            exec_fname_rel=f"main.{self.lang_ext}"
        )

        if results['correct']['correct']: 
            combined = results.get("metrics", {}).get("combined_score")
            db_program = Program(
                id=str(uuid.uuid4()),
                code=initial_code,
                parent_id=None,
                generation=0,
                code_diff="initial",
                correct=True,
                combined_score=combined,
                metadata={
                    "inference_time": 0.0,  # No inference time for generation 0
                    "compute_time": rtime,
                    "stdout_log": results.get("stdout_log", ""),
                    "stderr_log": results.get("stderr_log", ""),
                }
            )

            ray.get(self.db.add.remote(db_program, verbose=True))
        else:
            raise ValueError("Initial program is not correct. Please fix the initial program and try again.")
    