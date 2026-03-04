import io
import zipfile
import time
import uuid
import logging
import json
from typing import List, Optional, Union, cast
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
import ray
from rayevolve.core.dbase2 import ProgramDatabase, Program
from .worker2 import EvoWorker, EvoGen
from .common import EvolutionConfig, BackendConfig
from rayevolve.launch.ray_backend import RayExecutionBackend
import logfire

# Set up logging
logger = logging.getLogger(__name__)

## NOTE: EvolutionRunner still runs in the ray driver. 
class EvolutionRunner:
    def __init__(
        self,
        evo_config: EvolutionConfig,
        backend_config: BackendConfig,
        project_dir: str, 
        verbose: bool = False,
    ):
        self.evo_config = evo_config
        self.backend_config = backend_config
        self.project_dir = project_dir
        self.verbose = verbose

        # Get full path of results directory.
        if evo_config.results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = Path(f"results_{timestamp}").resolve()
        else:
            self.results_dir = Path(evo_config.results_dir).resolve()

        if self.verbose:
            logfire.configure()
            logger.addHandler(logfire.LogfireLoggingHandler())
            logger.setLevel(logging.INFO)

            # Also log the initial setup information
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Evolution run started at {start_time}")
            logger.info(f"Results directory: {self.results_dir}")

        # Check if we are resuming a run
        self.resuming_run = False
        self.start_gen = 0
        db_path = Path(f"{self.results_dir}/evolution_db.jsonl")
        
        initial_db_zip_bytes = None
        if self.evo_config.results_dir is not None and db_path.exists():
            self.resuming_run = True
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(db_path, arcname=db_path.name)
            initial_db_zip_bytes = buffer.getvalue()

        self.db = ProgramDatabase.remote(
            initial_db_zip_bytes=initial_db_zip_bytes
        )

        self.backend = RayExecutionBackend(
            config=backend_config,
            project_dir=self.project_dir,
            verbose=verbose,
        )
        
        if self.resuming_run:
            state_file = Path(f"{self.results_dir}/evogen_state.json")
            completed_generations = 0
            if state_file.exists():
                with open(state_file, "r") as f:
                    state_data = json.load(f)
                    completed_generations = state_data.get("generation", 0)
            else:
                logger.warning(f"Could not find {state_file}. Resuming from generation 0.")
                
            logger.info("RESUMING PREVIOUS EVOLUTION RUN")
            logger.info(
                f"Resuming evolution from: {self.results_dir}\n"
                f"Found {completed_generations} "
                "previously completed generations."
            )
            self.start_gen = completed_generations


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
                self.backend_config,
                self.backend.project_zip_bytes,
                self.db,
                self.verbose,
            )
            all_refs.append(worker.run.remote())

        cur_gen: int = ray.get(gen.get.remote()) 
        while cur_gen < self.evo_config.max_generations:
            zip_bytes: bytes = ray.get(self.db.download_database_zip.remote())
            if zip_bytes:
                with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
                    zf.extractall(self.results_dir)
            
            self._sync_zip_files()

            state_file = Path(f"{self.results_dir}/evogen_state.json")
            with open(state_file, "w") as f:
                json.dump({"generation": cur_gen}, f)

            time.sleep(self.evo_config.dl_evostate_freq)
            cur_gen = ray.get(gen.get.remote())

        # Now wait for ALL workers to finish
        ray.get(all_refs)

        # Download final database state
        zip_bytes: bytes = ray.get(self.db.download_database_zip.remote())
        if zip_bytes:
            with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
                zf.extractall(self.results_dir)

        self._sync_zip_files()

        # Save final generation state
        final_gen: int = ray.get(gen.get.remote())
        state_file = Path(f"{self.results_dir}/evogen_state.json")
        with open(state_file, "w") as f:
            json.dump({"generation": final_gen}, f)


    def _sync_zip_files(self):
        """Download any zip files from the DB that haven't been saved locally yet."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        zip_program_ids: List[str] = ray.get(self.db.get_zip_program_ids.remote())
        for pid in zip_program_ids:
            zip_path = self.results_dir / f"{pid}.zip"
            if not zip_path.exists():
                zip_bytes: bytes = ray.get(self.db.get_zip_bytes.remote(pid))
                zip_path.write_bytes(zip_bytes)

    def _run_generation_0(self):
        """Setup and run generation 0 to initialize the database."""
        if self.verbose:
            logger.info(
                f"Reading initial program from {self.project_dir}/{self.evo_config.evo_file}"
            )
        
        try:
            initial_code = Path(f"{self.project_dir}/{self.evo_config.evo_file}").read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Could not read initial program from {self.project_dir}/{self.evo_config.evo_file}. Error: {e}")

        # Run the evaluation code using the Ray backend.
        results, rtime, result_zip_bytes = self.backend.run_job(
            generated_code=initial_code,
            exec_fname_rel=self.evo_config.evo_file
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
                language=self.evo_config.lang_identifier,
                metadata={
                    "inference_time": 0.0,  # No inference time for generation 0
                    "compute_time": rtime,
                    "stdout_log": results.get("stdout_log", ""),
                    "stderr_log": results.get("stderr_log", ""),
                }
            )

            ray.get(self.db.add.remote(db_program, verbose=True))
            ray.get(self.db.add_zip_bytes.remote(db_program.id, result_zip_bytes))

            self.results_dir.mkdir(parents=True, exist_ok=True)
            zip_path = self.results_dir / f"{db_program.id}.zip"
            zip_path.write_bytes(result_zip_bytes)
        else:
            raise ValueError("Initial program is not correct. Please fix the initial program and try again.")
    