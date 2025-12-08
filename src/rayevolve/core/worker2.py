import random
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

@ray.remote
class EvoGen:
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


    async def run(self):
        #debugpy.listen(5678)
        #debugpy.wait_for_client()
        #debugpy.breakpoint()              
        while True:
            current_gen = ray.get(self.gen.next.remote())
            await self.run_strategy()

    async def run_strategy(self):            
        best_score_table = await self.db.get_best_score_table.remote() 

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

        evo_strategist = Agent(
            "google-gla:gemini-2.5-pro",
            output_type=StrategyProbs,
        )

        result = await evo_strategist.run(prompt)
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

        return await strategy_funcs[chosen_name]()

    async def agent_climb(self):
        elite_parent = await self.db.sample_archive_program.remote()
        if not elite_parent:
            return

        class StepStatus(str, Enum):
            pending = "pending"
            in_progress = "in_progress"
            completed = "completed"
            cancelled = "cancelled"

        class Step(BaseModel):
            status: StepStatus = Field(description="Current status of this step.")
            description: str = Field(description="Human-readable description of the step.")

        class ClimbContext(BaseModel):
            parent_program: str = Field(description="The original code of the elite parent program to improve.")
            program: str = Field(description="Current in progress improved program.")
            plan: List[Step] = Field(
                description="Ordered list of steps, each with a status and description."
            )

        class NoImprovement(BaseModel):
            reason: str = Field(description="Reason why no improvement was made.")

        evo_coder = Agent[ClimbContext, ImprovedProgram | NoImprovement](
            "google-gla:gemini-2.5-pro")
        
        @evo_coder.tool
        async def update_plan(ctx: RunContext[ClimbContext], new_plan: List[Step]) -> str:
            ctx.plan = new_plan
            return "Plan updated."

        @evo_coder.tool
        async def evaluate_program(program_code: str) -> str:
            return "Score: 0.95"  # Placeholder for actual evaluation logic

        @evo_coder.tool
        async def read_current_program() -> str:
            return elite_parent.code

        @evo_coder.tool
        async def apply_patch() -> str:
            return "Applied patch to the program."
        
        new_program = Program()

        await evo_coder.run(coder_prompt)
            await self.db.add.remote(new_program)


    async def agent_driftup(self):
        pass
    
    async def agent_driftaway(self):
        pass

    async def agent_jump(self):
        pass