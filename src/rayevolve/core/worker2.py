import random
from sys import stdout
import uuid
import time
import logging
import yaml
from rich.logging import RichHandler
from rich.table import Table
from rich.console import Console
import rich.box
from typing import List, Optional, Union, cast
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from subprocess import Popen
import ray
import asyncio
from rayevolve.launch import JobScheduler, JobConfig, ProcessWithLogging
from rayevolve.database import ProgramDatabase, DatabaseConfig, Program
from rayevolve.llm import (
    LLMClient,
    extract_between,
    EmbeddingClient,
    BanditBase,
    AsymmetricUCB,
)
from rayevolve.edit import (
    apply_diff_patch,
    apply_full_patch,
    summarize_diff,
    redact_immutable,
)
from rayevolve.core.sampler import PromptSampler
from .common import EvolutionConfig, RunningJob, FOLDER_PREFIX

import debugpy

import textwrap
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelMessage, RunContext, RunUsage, UsageLimits
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyProbs(BaseModel):
    """Probabilities for each evolutionary strategy."""
    climb: float = Field(..., description="Probability of choosing the climb strategy", ge=0)
    drift_up: float = Field(..., description="Probability of choosing the drift up strategy", ge=0)
    drift_away: float = Field(..., description="Probability of choosing the drift away strategy", ge=0)
    jump: float = Field(..., description="Probability of choosing the jump strategy", ge=0)
    reasoning: str = Field(..., description="A short sentence explaining the reasoning behind the chosen probabilities.")
    def as_normalized_weights(self) -> dict[str, float]:
        """Return a dict of normalized probabilities that sums to 1."""
        weights = {
            "climb": self.climb,
            "drift_up": self.drift_up,
            "drift_away": self.drift_away,
            "jump": self.jump,
        }
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("All probabilities are zero or negative")
        return {k: v / total for k, v in weights.items()}



class ExperimentEntry(BaseModel):
    idea: str = Field(description="Brief description of the experiment idea.")
    outcome: str = Field(description="Outcome or observation from running the experiment.")

class ClimbContext(BaseModel):
    parent_score: float = Field(description="Score of the parent program being improved.")
    experiment_log: List["ExperimentEntry"] = Field(
        default_factory=list,
        description="Ordered log of experiments conducted and their outcomes."
    )
    
class GiveUp(BaseModel):
    """Unable to improve the parent program."""

class ImprovedProgram(BaseModel):
    """Code of the improved program and its score."""
    improved_code: str = Field(description="The improved program code.")
    score: float = Field(description="The score of the improved program.")


@ray.remote
class EvoGen:
    """Keeps track of the current generation number."""
    def __init__(self, initial=0):
        self.generation = initial

    def get(self):
        return self.generation

    def set(self, value: int):
        self.generation = value

    def next(self):
        """Increment and return the new generation."""
        self.generation += 1
        return self.generation
    
@ray.remote
class EvoWorker:
    def __init__(self, 
                 worker_id: str,
                 gen: EvoGen,
                 evo_config: EvolutionConfig, 
                 job_config: JobConfig,
                 results_dir: str,
                 db: ProgramDatabase, 
                 verbose: bool):
        super().__init__()  
        self.worker_id = worker_id
        self.gen = gen
        self.evo_config = evo_config
        self.results_dir = results_dir
        self.db = db
        self.verbose = verbose

        self.scheduler = JobScheduler(
            job_type=evo_config.job_type,
            config=job_config,  # type: ignore
            verbose=verbose,
        )

        if evo_config.embedding_model is not None:
            self.embedding = EmbeddingClient(
                model_name=evo_config.embedding_model,
                verbose=verbose,
            )
        else:
            self.embedding = None

        # Initialize rich console for formatted output
        self.console = Console()

        if self.evo_config.language == "cuda":
            self.lang_ext = "cu"
        elif self.evo_config.language == "cpp":
            self.lang_ext = "cpp"
        elif self.evo_config.language == "python":
            self.lang_ext = "py"
        elif self.evo_config.language == "rust":
            self.lang_ext = "rs"
        else:
            msg = f"Language {self.evo_config.language} not supported"
            raise ValueError(msg)

    def run(self):
         
        while True:
            current_gen = ray.get(self.gen.next.remote())
            #await self.run_strategy(current_gen)
            self.agent_climb(current_gen)

    def run_strategy(self, current_gen: int):            
        best_score_table = ray.get(self.db.get_best_score_table.remote()) 

        template = textwrap.dedent("""
            You are the strategic supervisor for an evolutionary code optimization system.
            Your job is to set the probability distribution for the next worker based on the current progress trend.

            ### CURRENT STATUS (Best Score History)
            {best_score_table}

            "Time" is seconds since the epoch and "Best Score" is on an arbitrary scale where higher is better.

            ### ANALYSIS RULES (Focus on the most recent entries in the table)
            1. **Analyze Velocity:** Is the best score rising quickly, slowly, or flatlining?
            2. **Analyze Stagnation:**
            - If it is early in the simulation, stagnation cannot be definitively detected. Prioritize climbing until 
              there is evidence of stagnation.                      
            - Compare the current rate of improvement (or lack thereof) to previous periods of successful increase in the best score.
            - Determine if the current trend is significantly slower or has completely flattened compared to historical bests. 
              This establishes whether true stagnation or just slower growth is occurring.

            ### STRATEGY DEFINITIONS
            1. **CLIMB (Exploit):** Best when velocity is high and the score is improving.
            - Focuses on rigorously exploiting the current best solutions to achieve further, direct score improvements.
            2. **DRIFT UP (Exploit):** Best when velocity is slow.
            - Targets non-elite parents with potential, seeking to improve them and discover adjacent, potentially higher, peaks.
            3. **DRIFT AWAY (Explore):** Best when stuck short term.
            - Ignores score improvement. Tries to gradually change the approach while maintaining correctness, to escape local optima.
            4. **JUMP (Explore):** Best when stuck long term.
            - Generates fresh approaches that differ from current set of elites to explore new areas of the solution space.
        """)
        prompt = template.format(best_score_table=best_score_table)
        # Instructions for output format might not be necessary.
        """
            ### OUTPUT
            Return a JSON object with these keys. Ensure values sum to 1.0.
            - "reasoning": "A concise sentence explaining your strategic decision based on the historical data analysis."
            - "climb": float
            - "drift_up": float
            - "drift_away": float
            - "jump": float
        """

        evo_strategist = Agent("google-gla:gemini-2.5-pro", output_type=StrategyProbs)
        result = evo_strategist.run_sync(prompt)
        probs: StrategyProbs = result.output

        weights = probs.as_normalized_weights()

        # Map strategy names to the corresponding coroutine functions
        strategy_funcs = {
            "climb": self.agent_climb,
            "drift_up": self.agent_driftup,
            "drift_away": self.agent_driftaway,
            "jump": self.agent_jump,
        }

        chosen_name = random.choices(
            population=list(strategy_funcs.keys()),
            weights=[weights[name] for name in strategy_funcs.keys()],
            k=1,
        )[0]

        return strategy_funcs[chosen_name](current_gen)

    def agent_climb(self, current_gen: int):

        exec_fname = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/main.{self.lang_ext}"
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        elite_parent = ray.get(self.db.sample_archive_program.remote())
        if not elite_parent:
            return
        
        coder_template = textwrap.dedent("""
            Given the code of the following elite program:
            ```{lang}                             
            {code}
            ```
            This elite program achieved the following score:
            {score}

            Your task is to improve this program further. Make changes to the code
            to conduct experiments that improve its score.
            The improvements should aim to increase the score beyond {score}. 
            Keep track of your experiments and learn from them recording why you think each experiment succeeded or failed.
            Keep the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the code. Do not change the code outside of these markers.
            If after several attempts you cannot improve the score, you should give up.
        """)
        coder_prompt = coder_template.format(lang=self.evo_config.language, 
                                             code=elite_parent.code, 
                                             score=elite_parent.combined_score)


        evo_coder = Agent[ClimbContext, ImprovedProgram | GiveUp](
            "google-gla:gemini-2.5-flash")

        @evo_coder.tool
        def run_experiment(ctx: RunContext[ClimbContext], program: str) -> str:
            Path(exec_fname).write_text(program, "utf-8")
            start_time = time.time()
            job_id = self.scheduler.submit_async(exec_fname, results_dir)
            results = self.scheduler.get_job_results(job_id, results_dir)
            rtime = time.time() - start_time
             
            out_str = ""
            if results["correct"]:
                out_str += "The program executed correctly and produced a valid result.\n"
                out_str += f"It achieved a score of {results['metrics']['combined_score']}\n"
                if results['metrics']['combined_score'] > ctx.deps.parent_score:
                    out_str += f"This is an improvement over the parent program's score of {ctx.deps.parent_score}.\n"
                else:
                    out_str += f"However, this is not an improvement over the parent program's score of {ctx.deps.parent_score}.\n"
            else:
                out_str += "The program did not execute correctly and did not produce a valid result.\n"
                
            out_str += f"The evaluation took {rtime:.2f} seconds.\n"                
            out_str += "Here is the standard output of the program:\n"
            out_str += "```"
            out_str += results["stdout_log"] + "\n"
            out_str += "```\n"
            out_str += "Here is the standard error output of the program:\n"
            out_str += "```"
            out_str += results["stderr_log"] + "\n"
            out_str += "```\n"
            return out_str

        @evo_coder.tool
        def log_experiment(ctx: RunContext[ClimbContext], idea: str, outcome: str) -> str:
            ctx.deps.experiment_log.append(ExperimentEntry(idea=idea, outcome=outcome))
            log_str = "Experiments conducted so far:\n\n"
            for idx, entry in enumerate(ctx.deps.experiment_log):
                log_str += f"Experiment {idx+1}:\n"
                log_str += f"Idea: {entry.idea}\n"
                log_str += f"Outcome: {entry.outcome}\n\n"
            return log_str

        debugpy.listen(5678)
        debugpy.wait_for_client()
        debugpy.breakpoint()                 
        res = evo_coder.run_sync(coder_prompt, deps=ClimbContext(parent_score=elite_parent.combined_score))
        
        # Add the program to the database
        db_program = Program(
            id=str(uuid.uuid4()),
            code=res.code,
            language=self.evo_config.language,
            parent_id=elite_parent.parent_id,
            generation=current_gen,
            code_diff="agent_climb",
            embedding=[],
            correct=True,
            combined_score=res.score
        )
        ray.get(self.db.add.remote(db_program))


    def agent_driftup(self, current_gen: int):
        pass
    
    def agent_driftaway(self, current_gen: int):
        pass

    def agent_jump(self, current_gen: int):
        pass


    def get_code_embedding(self, exec_fname: str) -> tuple[List[float], float]:
        """Get the embedding of the code."""
        try:
            evaluated_code = Path(exec_fname).read_text(encoding="utf-8")
        except Exception as e:
            evaluated_code = ""
        if evaluated_code != "":
            # Get the embedding of the initial program
            try:
                if self.embedding is not None:
                    redacted_code = redact_immutable(evaluated_code, no_state=True)
                    embedding_result, e_cost = self.embedding.get_embedding(
                        redacted_code
                    )
                else:
                    embedding_result = []
                    e_cost = 0.0
                code_embedding = cast(List[float], embedding_result)
            except Exception as e:
                code_embedding = []
                e_cost = 0.0
        else:
            code_embedding = []
            e_cost = 0.0
        return code_embedding, e_cost
