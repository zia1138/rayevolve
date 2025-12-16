from __future__ import annotations
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
from typing import List, Optional, Union, cast, Tuple
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
from pydantic_ai import Agent, ModelRetry, ModelMessage, RunContext, RunUsage, UsageLimits
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

class ClimbContext(BaseModel):
    evolve_block: EvolveBlock
    parent_score: float

class DriftAwayContext(BaseModel):
    evolve_block: EvolveBlock

class VerifiedChange(BaseModel):
    """Use this when you can verify that the type of change provided was indeed made to the program."""

class ChangeNotVerified(BaseModel):
    """Use this when type of change provided was *NOT* made to the program."""
    reasoning: str = Field(description="Explanation of why the change type was not verified.")

class NovelProgram(BaseModel):
    """Use this when the provided program is indeed substantially different from the parent."""

class NotNovel(BaseModel):
    """Use this when the program is not substantially different from the parent."""
    reasoning: str = Field(description="Explanation of why the program is not novel.")

def clear_results_dir(results_dir: str) -> None:
    """
    Remove all files inside results_dir, keeping the folder itself.
    Safe to call if the directory does not exist.
    """
    p = Path(results_dir)
    if not p.exists():
        return
    for child in p.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete {child}: {e}")


class EvolveBlock(BaseModel):
    """
    Represents the dissected components of a source code file containing
    EVOLVE-BLOCK markers.
    """
    pre_block: str = Field(description="Content before the EVOLVE-BLOCK-START line.")
    start_marker_line: str = Field(description="The complete line containing the EVOLVE-BLOCK-START marker.")
    inner_content: str = Field(description="The original code content between the markers.")
    end_marker_line: str = Field(description="The complete line containing the EVOLVE-BLOCK-END marker.")
    post_block: str = Field(description="Content after the EVOLVE-BLOCK-END line.")
 
    def reconstruct(self, new_inner_content: str) -> str:
        """
        Reconstructs the full source code using these surrounding parts and
        the provided new inner content. Handles newline normalization.
        """
        # 1. Clean the new content (remove leading/trailing whitespace/newlines)
        cleaned_inner = new_inner_content.strip()
        
        # 2. Ensure the start marker line ends with a newline (safety check)
        start_line = self.start_marker_line
        if not start_line.endswith('\n') and not start_line.endswith('\r'):
            start_line += '\n'
 
        # 3. Format the inner block with a trailing newline if it has content
        formatted_inner = ""
        if cleaned_inner:
            formatted_inner = cleaned_inner + '\n'
 
        # 4. Concatenate
        return f"{self.pre_block}{start_line}{formatted_inner}{self.end_marker_line}{self.post_block}"

def extract_evolve_block(full_code: str) -> EvolveBlock:
    """
    Parses a source code string and returns a EvolveBlock object containing
    the separated components.
    
    Raises:
        ValueError: If markers are missing, duplicated, or out of order.
    """
    lines = full_code.splitlines(keepends=True)
    
    start_index = -1
    end_index = -1
     
    # Linear scan to find marker lines
    for i, line in enumerate(lines):
        if "EVOLVE-BLOCK-START" in line:
            if start_index != -1:
                raise ValueError("Multiple EVOLVE-BLOCK-START markers found.")
            start_index = i
        elif "EVOLVE-BLOCK-END" in line:
            if end_index != -1:
                raise ValueError("Multiple EVOLVE-BLOCK-END markers found.")
            end_index = i
             
    # Validation Logic
    if start_index == -1:
        raise ValueError("EVOLVE-BLOCK-START marker not found in code.")
    if end_index == -1:
        raise ValueError("EVOLVE-BLOCK-END marker not found in code.")
    if start_index >= end_index:
        raise ValueError(f"EVOLVE-BLOCK-START (line {start_index+1}) appears after or on same line as EVOLVE-BLOCK-END (line {end_index+1}).")
 
    # Construct the Pydantic Model
    return EvolveBlock(
        pre_block="".join(lines[:start_index]),
        start_marker_line=lines[start_index],
        inner_content="".join(lines[start_index+1 : end_index]),
        end_marker_line=lines[end_index],
        post_block="".join(lines[end_index+1:])
    )

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
            self.agent_driftaway(current_gen)
            # self.run_strategy(current_gen)
            #if random.random() < 0.5:
            #    self.agent_driftaway(current_gen)
            #else:
            #    if random.random() < 0.5:
            #        self.agent_climb(current_gen)
            #    else:
            #        self.agent_climb_or_drift(current_gen, drift_up=True)

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
        
        print("Current Generation:", current_gen)

        if drift_up:
            # Sample from non-elite programs for drift up.
            parent = ray.get(self.db.sample_all_programs.remote(10))
        else:        
            parent = ray.get(self.db.sample_archive_program.remote(3))

        evolve_block = extract_evolve_block(parent.code)

        coder_template = textwrap.dedent("""
            ### MISSION
            The code below has achieved a score of **{score}**. Your goal is to beat this score.
           
            ### CODE
            ```{lang}                             
            {code}
            ```
 
            ### PROTOCOL
            You must follow the **Scientific Method**:
            1. **Analyze:** Use the code for inspiration and come up with an a hypothesis on how to improve the score.
               It can be completely novel or a modification of the existing code.                     
            2. **Experiment:** Write the code to implement your idea.
            3. **Evaluate:** Use `run_experiment` to get the score.
             
            ### CONSTRAINTS
            - **Persistence:** Do not give up. Use the feedback to improve your code.
            - **Efficiency:** You have a maximum of 5 attempts.
            - **Interface:** Make sure your rewritten program maintains the same inputs and outputs as the original program, 
              but with an improved internal implementation.
            
            ### COMPLETION
            - As soon as you achieve a score greater than {score}, submit your improved program using `submit_tool`.
            - If you cannot beat the of the above code after 5 distinct hypotheses, give up and offer an explanation why.
         """)

        coder_prompt = coder_template.format(lang=self.evo_config.language, 
                                             code=evolve_block.inner_content, 
                                             score=parent.combined_score)

        def submit_tool(ctx: RunContext[ClimbContext], program: str) -> None:
            """
            Submit an improved program when you achieve a higher score relative to the parent program.
            Args:
                program: Code for the program that achieved a higher score.
            """            
            evo_program = ctx.deps.evolve_block.reconstruct(program)
            Path(exec_fname).write_text(evo_program, "utf-8")
            start_time = time.time()
            job_id = self.scheduler.submit_async(exec_fname, results_dir)
            results = self.scheduler.get_job_results(job_id, results_dir)
            rtime = time.time() - start_time

            if results.get("correct"): 
                combined = results.get("metrics", {}).get("combined_score")
                if combined is not None:
                    if combined > ctx.deps.parent_score:
                        # Add the program to the database
                        db_program = Program(
                            id=str(uuid.uuid4()),
                            code=evo_program,
                            language=self.evo_config.language,
                            parent_id=parent.id,
                            generation=current_gen,
                            code_diff="agent_climb",
                            embedding=[],
                            correct=True,
                            combined_score=combined,
                        )
                        ray.get(self.db.add.remote(db_program))
                    else:
                        clear_results_dir(results_dir)
                        raise ModelRetry("Improved program did not achieve a higher score on re-evaluation.")
                else:
                    clear_results_dir(results_dir)
                    raise ModelRetry("Improved program did not return a score on re-evaluation.")
            else:
                clear_results_dir(results_dir)
                raise ModelRetry("Improved program was not correct on re-evaluation.")

        model = GoogleModel('gemini-2.5-flash')
        settings = GoogleModelSettings(google_thinking_config={"thinking_budget":-1})

        evo_coder = Agent(
            model,
            system_prompt=self.evo_config.task_sys_msg,
            deps_type=ClimbContext,
            output_type=[str, submit_tool],
            retries=3,            
            model_settings=settings)

        @evo_coder.tool(retries = 3)
        def run_experiment(ctx: RunContext[ClimbContext], program: str, hypothesis: str) -> str:
            """
            Call this tool with a novel program that you want to evaluate. It will return
            the results of executing the program including its score and correctness.
            Args:
                program: A program that you think will achieve a higher score.
                hypothesis: A detailed description of the change being tested and why it should improve performance.
            Returns:
                str: A human-readable report of the results of the experiment.
            """
            Path(exec_fname).write_text(ctx.deps.evolve_block.reconstruct(program), "utf-8")
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

            # NOTE: This is an issue for any concurrency in this agent.
            clear_results_dir(results_dir)
            return out_str

        try:
            #debugpy.listen(5678)
            #debugpy.wait_for_client()
            #debugpy.breakpoint()
            evo_coder.run_sync(coder_prompt, 
                               deps=ClimbContext(evolve_block=evolve_block, 
                                                 parent_score=parent.combined_score))
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

        parent = ray.get(self.db.sample_all_programs.remote(3))
        evolve_block = extract_evolve_block(parent.code)

        coder_template = textwrap.dedent("""
            ### MISSION
            The code is below. Your goal is to produce a dramatically different solution that still works correctly.   
            You should make many different types of substantive changes including algorithmic, changes to
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
            - **Interface:** Make sure your rewritten program maintains the same inputs and outputs as the original program, 
              but with a completely novel internal implementation.
                                                      
            ### COMPLETION
            - As soon as the change type was verified and the program is substantially different from the parent, 
              use the submit_novel tool to submit your novel program.
            - If you cannot find a correct and novel program after 5 attempts explain why and give up.
         """)

        coder_prompt = coder_template.format(lang=self.evo_config.language, 
                                             code=evolve_block.inner_content)

        model = GoogleModel('gemini-2.5-flash')
        settings = GoogleModelSettings(google_thinking_config={"thinking_budget":-1})

        evo_change = Agent(model, output_type=VerifiedChange | ChangeNotVerified, model_settings=settings)
        evo_diff = Agent(model, output_type=NotNovel | NovelProgram, model_settings=settings)

        async def confirm_change(parent_code: str, novel_code: str, change_type: str) -> str:
            """
            Verify that the specified change type was applied to the novel program relative to the parent.

            Args:
                parent_code: The source code of the parent program.
                novel_code: The source code of the proposed novel program.
                change_type: A detailed description of the modification made (e.g., algorithmic change,
                    refactoring, control-flow alteration, etc.).

            Returns:
                A structured result indicating verification status:
                - VerifiedChange: The change type was confirmed and the program is substantially different.
                - ChangeNotVerified: The change type could not be confirmed or the program is not sufficiently novel,
                  with an explanation of why.
            """
            change_template = textwrap.dedent("""
            Given the the following parent program and new program
            Parent program:                                                                
            ```{lang}
            {parent_code}
            ```
            New program:                                
            ```{lang}
            {proposed_program}
            ```
            Verify that the the following change type was made:
            {change_type} 
            Return `VerifiedChange` if the change described above was made to the code.
            Return `ChangeNotVerified` if the program is not substantially different from the parent, explain why.
            """)
            change_prompt = change_template.format(parent_code=parent_code,
                                              proposed_program=novel_code,
                                              change_type=change_type,
                                              lang=self.evo_config.language)
            r = await evo_change.run(change_prompt)
            return r.output

        async def confirm_novelty(parent_code: str, novel_code: str) -> bool:
            """
            Evaluate whether the proposed program is substantially novel relative to the parent program.

            Args:
                parent_code: Source code of the parent program used as the baseline.
                novel_code: Source code of the proposed novel program to evaluate.

            Returns:
                A structured verdict produced by `evo_diff`:
                - VerifiedChangeAndNovel: The program exhibits substantial, meaningful differences.
                - NotNovel: The program is not sufficiently different, with reasoning.
            """            
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
            Does the new program have substantial changes including algorithmic, changes to
            modularization, data flow, control flow, or architectural patterns, or changes in parameters and parameterization.   
            from the parent program? Your bar should be high for what constitutes a substantial change.
            Return `NovelProgram` if the program is substantially different from the parent.
            Return `NotNovel` if the program is not substantially different from the parent and explain why. 
            """)
            diff_prompt = diff_template.format(parent_code=parent_code,
                                              proposed_program=novel_code,
                                              lang=self.evo_config.language)
            r = await evo_diff.run(diff_prompt)
            return r.output

        async def submit_novel(ctx: RunContext[DriftAwayContext], novel_program: str, change_type: str) -> None:
            """
            Submit a proposed novel program.
            Args:
                novel_program: A novel program that is dramatically different from the parent that 
                    still functions correctly.
                change_type: A detailed description of the modification made (e.g., algorithmic change,
                    refactoring, control-flow alteration, etc.).                
            """
            evo_program = ctx.deps.evolve_block.reconstruct(novel_program)
            Path(exec_fname).write_text(evo_program, "utf-8")
            start_time = time.time()
            job_id = self.scheduler.submit_async(exec_fname, results_dir)
            results = self.scheduler.get_job_results(job_id, results_dir)
            rtime = time.time() - start_time

            if results.get("correct"): 
                combined = results.get("metrics", {}).get("combined_score")
                if combined is not None:
                    confirmation = await confirm_change(ctx.deps.evolve_block.inner_content, novel_program, change_type)
                    novelty = await confirm_novelty(ctx.deps.evolve_block.inner_content, novel_program)
                    raise_str = ""
                    if isinstance(confirmation, ChangeNotVerified):
                        raise_str += "The change type could not be verified on re-evaluation. Here is the reason:\n"
                        raise_str += confirmation.reasoning + "\n"
                    if isinstance(novelty, NotNovel):
                        raise_str += "The program is not substantially different from the parent program on re-evaluation. Here is the reason:\n"
                        raise_str += novelty.reasoning + "\n"
                    if raise_str != "":
                        clear_results_dir(results_dir)
                        raise ModelRetry(raise_str)

                    # Add the program to the database
                    db_program = Program(
                        id=str(uuid.uuid4()),
                        code=evo_program,
                        language=self.evo_config.language,
                        parent_id=parent.id,
                        generation=current_gen,
                        code_diff="agent_driftaway",
                        embedding=[],
                        correct=True,
                        combined_score=combined,
                    )
                    ray.get(self.db.add.remote(db_program))
                else:
                    clear_results_dir(results_dir)
                    raise ModelRetry("Novel program did not return a score upon re-evaluation.")
            else:
                clear_results_dir(results_dir)
                raise ModelRetry("Novel program was not correct upon re-evaluation.")

        evo_coder = Agent(
            model,
            system_prompt=self.evo_config.task_sys_msg,
            deps_type=DriftAwayContext,
            output_type=[str, submit_novel],
            retries=3,
            model_settings=settings)

        @evo_coder.tool(retries = 3)
        async def check_novelty_and_correctness(ctx: RunContext[DriftAwayContext], novel_program: str, change_type:str) -> str:
            """
            Evaluate a proposed inner block for correctness and verify the described change type.

            Args:
                novel_program: A string containing a novel program that is dramtically different from the parent.
                change_type: A detailed description of the modification made (e.g., algorithmic change,
                    refactoring, control-flow alteration, etc.).                
            Returns:
                str: feedback for the agent including correctness, change verification, and novelty status.
            """
            evo_program = ctx.deps.evolve_block.reconstruct(novel_program)
            Path(exec_fname).write_text(evo_program, "utf-8")
            start_time = time.time()
            job_id = self.scheduler.submit_async(exec_fname, results_dir)
            results = self.scheduler.get_job_results(job_id, results_dir)
            rtime = time.time() - start_time

            out_str = ""
            if results.get("correct"):
                combined = results.get("metrics", {}).get("combined_score")
                if combined is not None:
                    out_str += f"The program executed correctly.\n"
                    confirmation = await confirm_change(ctx.deps.evolve_block.inner_content, novel_program, change_type)
                    novelty = await confirm_novelty(ctx.deps.evolve_block.inner_content, novel_program)

                    if isinstance(confirmation, VerifiedChange):
                        out_str += "The change type described was made to the program.\n"
                    elif isinstance(confirmation, ChangeNotVerified):
                        out_str += "The change type could not be verified. Here is the reason:\n"
                        out_str += confirmation.reasoning + "\n"
                    if isinstance(novelty, NovelProgram):
                        out_str += "The program is substantially different from the parent program.\n"
                    elif isinstance(novelty, NotNovel):
                        out_str += "The program is not substantially different from the parent program. Here is the reason:\n"
                        out_str += novelty.reasoning + "\n"
                else:
                    out_str += "The program did execute correctly and didn't produce a score.\n"
            else:
                out_str += "The program did not execute correctly and did not produce a valid score.\n"
        
            out_str += f"The evaluation took {rtime:.2f} seconds.\n"                
            out_str += "Here is the standard output of the program:\n"
            out_str += "```"
            out_str += results["stdout_log"] + "\n"
            out_str += "```\n"

            # NOTE: This is an issue for any concurrency in this agent.
            clear_results_dir(results_dir)            
            return out_str

        try:
            evo_coder.run_sync(coder_prompt, deps=DriftAwayContext(evolve_block=evolve_block))
        except Exception as e:
            print(f"Agent encountered an error: {e}")

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
