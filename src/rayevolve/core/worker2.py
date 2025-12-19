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
    """
    Probabilities and top-K pool size (beam width) for each evolutionary strategy. 
    exploit_weight and explore_weight must sum to 1. Keep exploit at a minimum of 0.3 to ensure steady progress.  
    """
    exploit_weight: float = Field(description="Probability [0.3 - 1.0]. Goal: Improve Score.")
    explore_weight: float = Field(description="Probability [0.0 - 0.7]. Goal: Novelty/Difference.")
    exploit_top_k: int = Field(description="Beam width for Exploitation")
    explore_top_k: int = Field(description="Beam width for Exploration")
    reasoning: str = Field(description="Analyze the trend velocity relative to historical difficulty. Explain your beam width adjustments based on the 'Catch the Mutants' philosophy.")
    def as_normalized_weights(self) -> dict[str, float]:
        """Return a dict of normalized probabilities that sums to 1."""
        weights = {
            "exploit_weight": self.exploit_weight,
            "explore_weight": self.explore_weight,
        }
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("All probabilities are zero or negative")
        return {k: v / total for k, v in weights.items()}

class ExploitContext(BaseModel):
    evolve_block: EvolveBlock
    parent_score: float

class ExploreContext(BaseModel):
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
            self.run_strategy(current_gen)
            
    def run_strategy(self, current_gen: int):            
        best_score_table = ray.get(self.db.get_best_score_table.remote()) 

        template = textwrap.dedent("""
        You are the Strategic Supervisor for an evolutionary optimization process.
        Your job is to tune the **Search Distribution** (Exploit vs. Explore) and **Beam Width** (Top-K) to match the current difficulty of the fitness landscape and the available compute resources.

        ### CURRENT STATUS
        - **Active Workers:** {num_workers} (Bandwidth)
        - **Total Programs:** {total_programs} (Population Depth)
        - **Best Score History:**
        {best_score_table}

        ### PHILOSOPHY: BEAM WIDTH & BANDWIDTH
        Your `Top-K` settings control the **Focus Intensity** (Ratio of Workers to Parents).
        1.  **Laser Focus (K=1):** All {num_workers} workers attack the same parent. Maximum depth, zero breadth.
        2.  **Balanced (K ~= Workers):** Roughly one worker per parent. Efficient parallel search.
        3.  **Wide Net (K > Workers):** Workers rotate through a large pool. Maximum breadth, low depth.

        ### DYNAMIC CONTROL RULES

        **1. BREAKOUT (The Snap)**
        - **Signal:** A new best score appears after a plateau.
        - **Action:** **SNAP THE BEAM SHUT.**
        - **Focus:** `exploit_top_k=1`.
        - **Reasoning:** We found a winner. Focus 100% of our {num_workers} workers on optimizing this single program immediately.

        **2. RISING PHASE (High Velocity)**
        - **Signal:** Frequent improvements relative to throughput.
        - **Action:** **NARROW FOCUS.**
        - **Focus:** `exploit_top_k=1` to `exploit_top_k=max(1, int({num_workers} * 0.2))`. Keep intensity high.

        **3. GRINDING PHASE (Decaying Velocity)**
        - **Signal:** Score is flat, but the duration is **comparable** to previous successful climbing intervals.
        - **Action:** **BROADEN THE BEAM (Balanced).**
        - **Focus:** `exploit_top_k` should match `Active Workers` ({num_workers}).
        - **Reasoning:** Optimization is noisy. If it usually takes 100 attempts to find a gain, do not panic at 100 failures. Keep grinding.

        **4. STAGNATION PHASE (The Wall)**
        - **Signal:** Zero improvement for a duration **significantly longer** than historical norms.
        - **Action:** **THE "GRADUAL SHIFT" MANEUVER.**
        - **Logic:** Shift resources from Exploit to Explore **proportionally** to the severity of the stagnation. As the plateau drags on, progressively increase `explore_weight` and widen the nets.
        - **Settings:**
            - **Exploit K:** Widen `exploit_top_k` to `2 * {num_workers}` (capped by `Total Programs`) to "Catch" mutants.
            - **Explore K:** Calibrate `explore_top_k` based on `Total Programs` and `Active Workers`.
            - For general structural change: `explore_top_k` should be `min({total_programs}, 2 * {num_workers})`.
            - For radical architectural rewrite of elite code: `explore_top_k` should be `max(1, int({num_workers} * 0.1))` (small, focused pool of elites).

        """)
        num_workers = 10
        total_programs = ray.get(self.db.total_programs.remote())
        prompt = template.format(best_score_table=best_score_table, num_workers=num_workers, total_programs=total_programs)
    
        evo_strategist = Agent(model='google-gla:gemini-2.5-pro', system_prompt=self.evo_config.task_sys_msg, output_type=StrategyProbs)
        result = evo_strategist.run_sync(prompt)
        probs: StrategyProbs = result.output

        weights = probs.as_normalized_weights()
        mode = random.choices(
            ["exploit", "explore"],
            weights=[weights["exploit_weight"], weights["explore_weight"]],
            k=1,
        )[0]

        if mode == "exploit":
            self.agent_exploit(current_gen, probs.exploit_top_k)
        else:
            self.agent_explore(current_gen, probs.explore_top_k)


    def agent_exploit(self, current_gen: int, parent_selection_top_k: int):
        exec_fname = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/main.{self.lang_ext}"
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        parent = ray.get(self.db.sample_all_topK.remote(parent_selection_top_k))
    
        evolve_block = extract_evolve_block(parent.code)

        exploit_template = textwrap.dedent("""
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

        exploit_prompt = exploit_template.format(lang=self.evo_config.language, 
                                                code=evolve_block.inner_content, 
                                                score=parent.combined_score)

        def submit_tool(ctx: RunContext[ExploitContext], program: str) -> None:
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
                            code_diff="agent_exploit",
                            correct=True,
                            combined_score=combined,
                        )
                        ray.get(self.db.add.remote(db_program))
                    else:
                        clear_results_dir(results_dir)
                        raise ModelRetry("Improved program did not achieve a higher score on submission.")
                else:
                    clear_results_dir(results_dir)
                    raise ModelRetry("Improved program did not return a score on submission.")
            else:
                clear_results_dir(results_dir)
                raise ModelRetry("Improved program was not correct on submission.")

        model = GoogleModel('gemini-2.5-flash')
        settings = GoogleModelSettings(google_thinking_config={"thinking_budget":-1})

        evo_exploit = Agent(
            model,
            system_prompt=self.evo_config.task_sys_msg,
            deps_type=ExploitContext,
            output_type=[str, submit_tool],
            retries=3,            
            model_settings=settings)

        @evo_exploit.tool(retries = 3)
        def run_experiment(ctx: RunContext[ExploitContext], program: str, hypothesis: str) -> str:
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
            evo_exploit.run_sync(exploit_prompt, 
                               deps=ExploitContext(evolve_block=evolve_block, 
                                                 parent_score=parent.combined_score))
        except Exception as e:
            print(f"Agent encountered an error: {e}")
    
    def agent_explore(self, current_gen: int, parent_selection_top_k: int):             
        exec_fname = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/main.{self.lang_ext}"
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        parent = ray.get(self.db.sample_all_topK.remote(parent_selection_top_k))

        evolve_block = extract_evolve_block(parent.code)

        explore_template = textwrap.dedent("""
            ### MISSION
            The code is below. Your goal is to produce a dramatically different solution that still works correctly.   
            You should make many different types of substantive changes including conceptual, algorithmic, changes to
            modularization, data flow, control flow, or architectural patterns, or changes in parameters
            and parameterization.                                                                  
           
            ### CODE
            ```{lang}                             
            {code}
            ```
                                         
            ### PROTOCOL
            You must propose a different solution and determine it is correct. You can submit 
            your solution to `check_novelty_and_correctness` with the change type you made and get feedback.
             
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

        explore_prompt = explore_template.format(lang=self.evo_config.language, 
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
                change_type: A detailed description of the modification made (e.g. conceptual, algorithmic change,
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
            Return `ChangeNotVerified` if the change described above was not made or cannot be confirmed, explain why.
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
            Does the new program have substantial changes including conceptual, algorithmic, changes to
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

        async def submit_novel(ctx: RunContext[ExploreContext], novel_program: str, change_type: str) -> None:
            """
            Submit a proposed novel program.
            Args:
                novel_program: A novel program that is dramatically different from the parent that 
                    still functions correctly.
                change_type: A detailed description of the modification made (e.g. , conceptual, algorithmic change,
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
                        raise_str += "The change type could not be verified on submission. Here is the reason:\n"
                        raise_str += confirmation.reasoning + "\n"
                    if isinstance(novelty, NotNovel):
                        raise_str += "The program is not substantially different from the parent program on submission. Here is the reason:\n"
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
                        code_diff="agent_explore",
                        embedding=[],
                        correct=True,
                        combined_score=combined,
                    )
                    ray.get(self.db.add.remote(db_program))
                else:
                    clear_results_dir(results_dir)
                    raise ModelRetry("Novel program did not return a score on submission.")
            else:
                clear_results_dir(results_dir)
                raise ModelRetry("Novel program was not correct on submission.")

        evo_explore = Agent(
            model,
            system_prompt=self.evo_config.task_sys_msg,
            deps_type=ExploreContext,
            output_type=[str, submit_novel],
            retries=3,
            model_settings=settings)

        @evo_explore.tool(retries = 3)
        async def check_novelty_and_correctness(ctx: RunContext[ExploreContext], novel_program: str, change_type:str) -> str:
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
            evo_explore.run_sync(explore_prompt, deps=ExploreContext(evolve_block=evolve_block))
        except Exception as e:
            print(f"Agent encountered an error: {e}")


