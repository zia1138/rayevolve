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
import logfire
from enum import Enum

from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

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
    """Use this when you are unable to improve the parent program."""

class ImprovedProgram(BaseModel):
    """Use this when you have successfully improved the parent program in one of your experiments."""
    improved_code: str = Field(description="The improved program code.")
    score: float = Field(description="The score of the improved program.")

class DriftContext(BaseModel):
    parent_code: str = Field(description="Code of the parent program being modified.")

class NovelProgram(BaseModel):
    """Use this when you have discovered a novel and correct program."""
    novel_code: str = Field(description="Novel and correct program code.")

class VerifiedChangeAndNovel(BaseModel):
    """Use this when you can verify the change type and the program is novel."""
    reasoning: str = Field(description="Explanation of the verification and novelty.")

class ChangeNotVerified(BaseModel):
    """Use this when the change type could not be verified."""
    reasoning: str = Field(description="Explanation of why the change type could not be verified.")

class NotNovel(BaseModel):
    """Use this when the program is not substantially different from the parent."""
    reasoning: str = Field(description="Explanation of why the program is not novel.")


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
        
        logfire.configure()
        logfire.instrument_pydantic_ai()

    def run(self):
        #debugpy.listen(5678)
        #debugpy.wait_for_client()
        #debugpy.breakpoint()                     
        while True:
            current_gen = ray.get(self.gen.next.remote())
            # self.run_strategy(current_gen)
            if random.random() < 1.0:
                self.agent_driftaway(current_gen)
            else:
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

    def agent_climb_or_drift(self, current_gen: int, drift_up: bool = False):
        exec_fname = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/main.{self.lang_ext}"
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        if drift_up:
            elite_parent = ray.get(self.db.sample_all_programs.remote())
        else:        
            elite_parent = ray.get(self.db.sample_archive_program.remote())

        coder_template = textwrap.dedent("""
            You are an expert computer scientist improving a solution.
            
            ### MISSION
            The code below has achieved a score of **{score}**. Your goal is to beat this score.
           
            ### CODE
            ```{lang}                             
            {code}
            ```
 
            ### PROTOCOL
            You must follow the **Scientific Method**:
            1. **Analyze:** Use the code for inspiration and come up with an approach to improve the score.
               It can be completely novel or a modification of the existing code.                     
            2. **Experiment:** Write the code to implement your idea.
            3. **Evaluate:** Use `run_experiment` to get the score.
            4. **Log:** Use `log_experiment` to record the outcome of your idea and analyze *why* 
               the score improved or got worse to inform your next attempt.
             
            ### CONSTRAINTS
            - **Persistence:** Do not give up. Use the feedback to improve your code.
            - **Efficiency:** You have a maximum of 5 attempts.
            - You must `run_experiment` before `log_experiment` for each hypothesis.                             
            - **Safety:** You may only modify code that lies below a line containing "EVOLVE-BLOCK-START" 
              and above a line containing "EVOLVE-BLOCK-END". Everything outside those markers is read-only 
              and must be kept as-is.                                         

            ### COMPLETION
            - If you achieve `new_score > {score}`. Stop and return `ImprovedProgram` with your best code and score.                                                                      
            - If you cannot beat the score after 5 distinct hypotheses, return `GiveUp`.
         """)

        coder_prompt = coder_template.format(lang=self.evo_config.language, 
                                             code=elite_parent.code, 
                                             score=elite_parent.combined_score)

        model = GoogleModel('gemini-2.5-flash')
        settings = GoogleModelSettings(google_thinking_config={"thinking_budget":-1})

        evo_coder = Agent[ClimbContext, ImprovedProgram | GiveUp](
            model,
            system_prompt=self.evo_config.task_sys_msg,
            deps_type=ClimbContext,
            output_type= ImprovedProgram | GiveUp,
            model_settings=settings)

        @evo_coder.tool
        def run_experiment(ctx: RunContext[ClimbContext], program: str) -> str:
            """Call this tool with a novel program that you want to evaluate. It will return
            the results of executing the program including its score and correctness.
            """
            Path(exec_fname).write_text(program, "utf-8")
            start_time = time.time()
            job_id = self.scheduler.submit_async(exec_fname, results_dir)
            results = self.scheduler.get_job_results(job_id, results_dir)
            rtime = time.time() - start_time
             
            out_str = ""
            if results.get("correct"):
                out_str += "The program executed correctly and produced a valid result.\n"
                combined = results.get("metrics", {}).get("combined_score")
                if combined is not None:
                    out_str += f"It achieved a score of {combined}\n"
                    if combined > ctx.deps.parent_score:
                        out_str += f"This is an improvement over the parent program's score of {ctx.deps.parent_score}.\n"
                    else:
                        out_str += f"However, this is not an improvement over the parent program's score of {ctx.deps.parent_score}.\n"
                else:
                    out_str += "Something happened and the score was not available in results.\n"
            else:
                out_str += "The program did not execute correctly and did not produce a valid result.\n"
        
            out_str += f"The evaluation took {rtime:.2f} seconds.\n"                
            out_str += "Here is the standard output of the program:\n"
            out_str += "```"
            out_str += results["stdout_log"] + "\n"
            out_str += "```\n"
            return out_str

        @evo_coder.tool
        def log_experiment(ctx: RunContext[ClimbContext], idea: str, outcome: str) -> str:
            """Call this tool after each experiment to log your idea and its outcome.
            This will help you keep track of what you've tried and what worked or didn't work."""
            ctx.deps.experiment_log.append(ExperimentEntry(idea=idea, outcome=outcome))
            log_str = "Experiments conducted so far:\n\n"
            for idx, entry in enumerate(ctx.deps.experiment_log):
                log_str += f"Experiment {idx+1}:\n"
                log_str += f"Idea: {entry.idea}\n"
                log_str += f"Outcome: {entry.outcome}\n\n"
            return log_str

        #debugpy.listen(5678)
        #debugpy.wait_for_client()
        #debugpy.breakpoint()        
        try:
            agent_result = evo_coder.run_sync(coder_prompt, deps=ClimbContext(parent_score=elite_parent.combined_score))
            
            if isinstance(agent_result.output, ImprovedProgram):
                Path(exec_fname).write_text(agent_result.output.improved_code, "utf-8")
                start_time = time.time()
                job_id = self.scheduler.submit_async(exec_fname, results_dir)
                results = self.scheduler.get_job_results(job_id, results_dir)
                rtime = time.time() - start_time
                if results.get("correct"): 
                    combined = results.get("metrics", {}).get("combined_score")
                    if combined is not None:
                        # Add the program to the database
                        db_program = Program(
                            id=str(uuid.uuid4()),
                            code=agent_result.output.improved_code,
                            language=self.evo_config.language,
                            parent_id=elite_parent.id,
                            generation=current_gen,
                            code_diff="agent_climb",
                            embedding=[],
                            correct=True,
                            combined_score=agent_result.output.score,
                        )
                        ray.get(self.db.add.remote(db_program))
                    else:
                        print("Improved program did not return a score upon re-evaluation.")
                else:
                    print("Improved program was not correct upon re-evaluation.")
        except Exception as e:
            print(f"Agent encountered an error: {e}")

    def agent_climb(self, current_gen: int):
        return self.agent_climb_or_drift(current_gen, drift_up=False)
    
    def agent_driftup(self, current_gen: int):
        return self.agent_climb_or_drift(current_gen, drift_up=True)
    
    def agent_driftaway(self, current_gen: int):             
        exec_fname = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/main.{self.lang_ext}"
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        elite_parent = ray.get(self.db.sample_all_programs.remote())

        coder_template = textwrap.dedent("""
            You are an expert computer scientist finding novel solutions.
            
            ### MISSION
            The code is below. Your goal is to produce a different solution that still works correctly.   
            You can make many different types of substantive changes including algorithmic, changes to
            modularization, data flow, control flow, or architectural patterns, or changes in parameters
            and parameterization.                                                                  
           
            ### CODE
            ```{lang}                             
            {code}
            ```
                                         
            ### PROTOCOL
            You must propose a different solution and determine it is correct. You can submit 
            your solution to `check_correctness` with the change type you made and get feedback.
             
            ### CONSTRAINTS
            - **Persistence:** Do not give up. Use the feedback to help you identify a novel approach.
            - **Efficiency:** You have a maximum of 5 attempts.
            - **Safety:** You may only modify code that lies below a line containing "EVOLVE-BLOCK-START" 
              and above a line containing "EVOLVE-BLOCK-END". Everything outside those markers is read-only 
              and must be kept as-is.
                                         
            ### COMPLETION
            - If you succeed in finding a correct and novel program return `NovelProgram` with your code.
            - If you cannot find a correct and novel program after 5 attempts, return `GiveUp`.
         """)

        coder_prompt = coder_template.format(lang=self.evo_config.language, 
                                             code=elite_parent.code)

        model = GoogleModel('gemini-2.5-flash')
        settings = GoogleModelSettings(google_thinking_config={"thinking_budget":-1})

        evo_coder = Agent[DriftContext, NovelProgram | GiveUp](
            model,
            system_prompt=self.evo_config.task_sys_msg,
            deps_type=DriftContext,
            output_type=NovelProgram | GiveUp,
            model_settings=settings)

        evo_diff = Agent(model, output_type=VerifiedChangeAndNovel | NotNovel | ChangeNotVerified,
                         model_settings=settings)

        def confirm_change(parent_code: str, novel_code: str, change_type: str) -> str:
            diff_template = textwrap.dedent("""
            Given the the following parent program and new program
            Parent program:                                                                
            ```{lang}
            {parent_code}
            ```
            New program:                                
            ```{lang}
            {proposed_program}
            ```
            1. Verify that the the following change type was made:
            {change_type} 
            2. Does this change result in a program that has substantial changes including algorithmic, changes to
               modularization, data flow, control flow, or architectural patterns, or changes in parameters and parameterization.   
               from the parent program?
            Return `VerifiedChangeAndNovel` if the change type could be verified and the program is substantially different.
            Return `ChangeNotVerified` if the change type could not be verified.
            Return `NotNovel` if the program is not substantially different from the parent.                                            
            """)
            diff_prompt = diff_template.format(parent_code=parent_code,
                                              proposed_program=novel_code,
                                              change_type=change_type,
                                              lang=self.evo_config.language)
            r = evo_diff.run(diff_prompt)
            return r.output             

        @evo_coder.tool
        def check_correctness(ctx: RunContext[DriftContext], program: str, change_type:str) -> str:
            """Call this tool with a novel program that you want to check for correctness and a description
            of the change type you made. It will return the results of executing the program including its correctness 
            It will check that the change type you provided was applied and if the program is substantially different from the parent.
            """
            debugpy.listen(5678)
            debugpy.wait_for_client()
            debugpy.breakpoint()   
            Path(exec_fname).write_text(program, "utf-8")
            start_time = time.time()
            job_id = self.scheduler.submit_async(exec_fname, results_dir)
            results = self.scheduler.get_job_results(job_id, results_dir)
            rtime = time.time() - start_time
             
            out_str = ""
            if results.get("correct"):
                combined = results.get("metrics", {}).get("combined_score")
                if combined is not None:
                    out_str += f"The program executed correctly.\n"
                    confirmation = confirm_change(ctx.deps.parent_code, program, change_type)
                    if isinstance(confirmation, VerifiedChangeAndNovel):
                        out_str += "The change type was verified and the program is substantially different from the parent.\n"
                    elif isinstance(confirmation, ChangeNotVerified):
                        out_str += "The change type could not be verified.\n"
                    elif isinstance(confirmation, NotNovel):
                        out_str += "The program is not substantially different from the parent.\n"
                else:
                    out_str += "The program did not execute correctly.\n"
            else:
                out_str += "The program did not execute correctly and did not produce a valid result.\n"
        
            out_str += f"The evaluation took {rtime:.2f} seconds.\n"                
            out_str += "Here is the standard output of the program:\n"
            out_str += "```"
            out_str += results["stdout_log"] + "\n"
            out_str += "```\n"
            return out_str

        try:
            agent_result = evo_coder.run_sync(coder_prompt, deps=DriftContext(parent_code=elite_parent.code))
            
            if isinstance(agent_result.output, ImprovedProgram):
                Path(exec_fname).write_text(agent_result.output.improved_code, "utf-8")
                job_id = self.scheduler.submit_async(exec_fname, results_dir)
                results = self.scheduler.get_job_results(job_id, results_dir)
                if results.get("correct"): 
                    combined = results.get("metrics", {}).get("combined_score")
                    if combined is not None:
                        # Add the program to the database
                        db_program = Program(
                            id=str(uuid.uuid4()),
                            code=agent_result.output.improved_code,
                            language=self.evo_config.language,
                            parent_id=elite_parent.id,
                            generation=current_gen,
                            code_diff="agent_climb",
                            embedding=[],
                            correct=True,
                            combined_score=agent_result.output.score,
                        )
                        ray.get(self.db.add.remote(db_program))
                    else:
                        print("Program did not return a score upon re-evaluation.")
                else:
                    print("Program was not correct upon re-evaluation.")
        except Exception as e:
            print(f"Agent encountered an error: {e}")


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
