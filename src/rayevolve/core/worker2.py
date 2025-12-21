from __future__ import annotations
import random
import uuid
import time
import logging
from pathlib import Path
from subprocess import Popen
import ray

from rayevolve.launch import JobScheduler, JobConfig, ProcessWithLogging
from rayevolve.database import ProgramDatabase, DatabaseConfig, Program
from .common import EvolutionConfig, RunningJob, FOLDER_PREFIX

import debugpy
import textwrap

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, ModelMessage, RunContext, RunUsage, UsageLimits
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

import logfire
import importlib.metadata

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
    explore_performance_floor: float = Field(description="Minimum relative score (0.0-1.0) to accept a novel program during exploration.")
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
    inference_start: float

class ExploreContext(BaseModel):
    evolve_block: EvolveBlock
    floor_score: float
    inference_start: float
    run_experiment_count: int = 0
    list_package_count: int = 0
    inspiration_count: int = 0

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
    Represents the dissected components of a source code file containing EVOLVE-BLOCK markers.
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

        logfire.configure(scrubbing=False)
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
        Your job is to tune the **Search Distribution** (Exploit vs. Explore) and **Beam Width** (Top-K) to match the 
        current difficulty of the fitness landscape and the available compute resources.

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
        - **Reasoning:** We found a winner. Focus all of our {num_workers} workers on optimizing this single program immediately.
        - **Explore Floor:** explore_performance_floor recommendation 0.95. High standards. Do not accept regressions during any exploration.                                   

        **2. RISING PHASE (High Velocity)**
        - **Signal:** Frequent improvements relative to throughput.
        - **Action:** **NARROW FOCUS.**
        - **Focus:** `exploit_top_k=1` to `exploit_top_k=max(1, int({num_workers} * 0.2))`. Keep intensity high.
        - **Explore Floor:** explore_performance_floor recommendation 0.90. Maintain momentum.                                   
                                   
        **3. GRINDING PHASE (Decaying Velocity)**
        - **Signal:** Score is flat, but the duration is **comparable** to previous successful climbing intervals.
        - **Action:** **BROADEN THE BEAM (Balanced).**
        - **Focus:** `exploit_top_k` should match `Active Workers` ({num_workers}).
        - **Reasoning:** Optimization is noisy. Keep grinding.
        - **Explore Floor:** explore_performance_floor recommendation 0.80. Allow moderate variance to find local improvements.                                   

        **4. STAGNATION PHASE (The Wall)**
        - **Signal:** Zero improvement for a duration **significantly longer** than historical norms.
        - **Action:** **THE "GRADUAL SHIFT" MANEUVER.**
        - **Logic:** Shift resources from Exploit to Explore **proportionally** to the severity of the stagnation. 
          As the plateau drags on, progressively increase `explore_weight` and widen the nets.
        - **Inference Time:** The table above shows inference time in seconds. It will get longer as programs that improve score get harder
          to find. Take this into account before switching into stagnation mode. Avoid premature stagnation if inference times are increasing. 
        - **Settings:**
            - **Exploit K:** Widen `exploit_top_k` to `2 * {num_workers}` (capped by `Total Programs`) to "Catch" mutants.
            - **Explore K:** Calibrate `explore_top_k` based on `Total Programs` and `Active Workers`.
            - **Explore Floor:** Gradually drop explore_performance_floor to about 0.60. Desperate times. Allow significant temporary regressions to find new architectural hills.                        
            - For general structural change: `explore_top_k` should be `min({total_programs}, 2 * {num_workers})`.
            - For radical architectural rewrite of elite code: `explore_top_k` should be `max(1, int({num_workers} * 0.1))` (small, focused pool of elites).
        """)
        num_workers = 11
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
            self.agent_exploit(current_gen, probs.exploit_top_k, probs.explore_top_k)
        else:
            self.agent_explore(current_gen, probs.explore_top_k, probs.explore_performance_floor)


    def agent_exploit(self, current_gen: int, parent_selection_top_k: int, explore_top_k: int):
        exec_fname = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/main.{self.lang_ext}"
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        parent = ray.get(self.db.sample_all_topK.remote(parent_selection_top_k))
        evolve_block = extract_evolve_block(parent.code)
        inference_start = time.time()

        exploit_template = textwrap.dedent("""
            ### MISSION
            The code below has achieved a score of **{score}**. Your goal is to beat this score.
           
            ### CODE
            ```python                             
            {code}
            ```
 
            ### PROTOCOL
            You must follow the **Scientific Method**:
            1. **Analyze:** Analyze the code above and come up with an approach to improve the score.
            2. **Packages** Use any imported packages to help you implement your ideas for improving the score.
            2. **Innovate:** Focus on substantial algorithmic changes and rewrites that can lead to significant improvements.                            
            3. **Experiment:** Write the code to implement your idea.
               You are encouraged to add print statements to help you debug and understand the code behavior.
            4. **Evaluate:** Use `run_experiment` to get any output and the score.
            5. **Inspiration:** Use `get_inspiration` to get concepts from other successful programs.  
             
            ### CONSTRAINTS
            - **Persistence:** Do not give up. Use the feedback to identify new approaches for improvement.
            - **Efficiency:** You have a maximum of 5 attempts.
            - **Interface:** Make sure your rewritten program maintains the same inputs and outputs as the original program, 
              but with an improved internal implementation.
            - **Originality:** If you use `get_inspiration`, do NOT copy the code verbatim. Adapt the **concepts** to your specific context.
            
            ### COMPLETION
            - As soon as you achieve a score greater than {score}, submit your improved program using `submit`.
            - If you cannot beat the score of the above code after 5 distinct hypotheses, give up and offer an explanation why.
         """)

        exploit_prompt = exploit_template.format(code=evolve_block.inner_content, score=parent.combined_score)

        model = GoogleModel('gemini-2.5-flash')
        settings = GoogleModelSettings(google_thinking_config={"thinking_budget":-1})

        def submit(ctx: RunContext[ExploitContext], program: str) -> None:
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
            
            if results.get("correct", False): 
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
                            metadata={
                                "inference_time": time.time() - ctx.deps.inference_start,
                                "compute_time": rtime,
                            }
                        )
                        ray.get(self.db.add.remote(db_program))
                    else:
                        clear_results_dir(results_dir)
                        raise ModelRetry("Improved program did not achieve a higher score on submission. Analyze why it failed and fix the issue.")
                else:
                    clear_results_dir(results_dir)
                    raise ModelRetry("Improved program did not return a score on submission. Analyze why this happened and fix the issue.")
            else:
                clear_results_dir(results_dir)
                raise ModelRetry("Improved program was not correct on submission. Analyze why this happened and fix the issue. Here is the error:", results.get("error", "Unknown Error"))

        evo_exploit = Agent(
            model,
            system_prompt=self.evo_config.task_sys_msg,
            deps_type=ExploitContext,
            output_type=[str, submit],
            retries=3,
            model_settings=settings)

        @evo_exploit.tool(retries = 3)
        def run_experiment(ctx: RunContext[ExploitContext], program: str, change: str) -> str:
            """
            Call this tool with an improved program that you want to evaluate. It will return
            the results of executing the program including its score, correctness, and stdout/stderr.
            Args:
                program: A program that you think will achieve a higher score.
                change: A detailed description of the change being tested and why it should improve performance.
            Returns:
                str: A human-readable report of the results of the experiment.
            """
            Path(exec_fname).write_text(ctx.deps.evolve_block.reconstruct(program), "utf-8")
            start_time = time.time()
            job_id = self.scheduler.submit_async(exec_fname, results_dir)
            results = self.scheduler.get_job_results(job_id, results_dir)
            rtime = time.time() - start_time

            out_str = ""
            if results.get("correct", False):
                out_str += "The program executed correctly and produced a valid result.\n"
                combined = results.get("metrics", {}).get("combined_score")
                if combined is not None:
                    out_str += f"It achieved a score of {combined}\n"
                    if combined > ctx.deps.parent_score:
                        out_str += f"This is an improvement over the parent program's score of {ctx.deps.parent_score}. You can now use `submit`.\n"
                    else:
                        out_str += f"However, this is not an improvement over the parent program's score of {ctx.deps.parent_score}. Analyze why it failed and propose a better approach.\n"
                else:
                    out_str += "Something happened and the score was not available in results. Analyze why this happened and propose a fix.\n"
            else:
                out_str += "The program did not execute correctly and did not produce a valid result. Analyze why this happened and propose a fix.\n"
                out_str += "Here is the error: " + results.get("error", "Unknown Error") + "\n"
        
            out_str += f"The evaluation took {rtime:.2f} seconds.\n"                
            out_str += "Here is the standard output of the program:\n"
            out_str += "```"
            out_str += results["stdout_log"] + "\n"
            out_str += "```\n"
            out_str += "Here is the standard error of the program:\n"
            out_str += "```"
            out_str += results["stderr_log"] + "\n"
            out_str += "```\n"
            # NOTE: This is an issue for any concurrency in this agent.
            clear_results_dir(results_dir)
            return out_str


        try:
            exploit_ctx = ExploitContext(evolve_block=evolve_block, 
                                         parent_score=parent.combined_score,
                                         inference_start=inference_start)
            evo_exploit.run_sync(exploit_prompt, deps=exploit_ctx)
        except Exception as e:
            print(f"Agent encountered an error: {e}")
    
    def agent_explore(self, current_gen: int, parent_selection_top_k: int, explore_performance_floor: float):            
        exec_fname = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/main.{self.lang_ext}"
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        parent = ray.get(self.db.sample_all_topK.remote(parent_selection_top_k))
        inference_start = time.time()

        evolve_block = extract_evolve_block(parent.code)
        floor_score = parent.combined_score * explore_performance_floor
        explore_template = textwrap.dedent("""
            ### MISSION
            The code below has achieved a score of **{score}**. 
            Your goal is to produce a dramatically different solution that still works correctly and
            achieves a score of **greater than {floor_score}**.
            You should consider different types of substantive changes including use of external packages, 
            conceptual, algorithmic, changes to modularization, data flow, control flow, or architectural patterns, or changes in parameters
            and parameterization, etc.
           
            ### CODE
            ```python
            {code}
            ```
                                         
            ### PROTOCOL
            You must propose a different solution and determine it is correct. You can submit 
            your solution to `run_experiment` with a description of the changes you made to get feedback.
            Identify helpful packages that will lead to creative solutions. You can use `list_packages` to see what is installed and
            import these packages in your code. You can also use `install_package` to install new packages.                                           
             
            ### CONSTRAINTS
            - **Persistence:** Do not give up. Use the feedback to help you identify a novel approach.
            - **Efficiency:** You have a maximum of 5 attempts.
            - **Interface:** Make sure your rewritten program maintains the same inputs and outputs as the original program, 
              but with a completely novel internal implementation.
            - **Originality:** If you use `get_inspiration`, do NOT copy the code verbatim. Adapt the **concepts** to your specific context. 
                                           
            ### COMPLETION
            - Make sure you perform research using the `list_packages` tool to identify useful pacakges that help you find novel solutions.
            - Make sure you perform research usign the `get_inspiration` tool to identify useful concepts from other successful programs.    
            - Once you achieve a score greater than {floor_score}, use the `submit` tool to submit your novel program.
            - If you cannot find a correct, novel program that achieves the minimum score, after 5 attempts explain why and give up.
        """)

        explore_prompt = explore_template.format(code=evolve_block.inner_content,
                                                score=parent.combined_score,
                                                floor_score=floor_score)

        model = GoogleModel('gemini-2.5-flash')
        settings = GoogleModelSettings(google_thinking_config={"thinking_budget":-1})
        #explore_packages = Agent(model, output_type=VerifiedChange | ChangeNotVerified, model_settings=settings)

        async def submit(ctx: RunContext[ExploreContext], novel_program: str, change: str) -> None:
            """
            Submit a proposed novel program.
            Args:
                novel_program: A novel program that is dramatically different from the parent that 
                    still functions correctly.
                change: A detailed description of the modification made (e.g. , conceptual, algorithmic change,
                    refactoring, control-flow alteration, etc.). 
            """
            evo_program = ctx.deps.evolve_block.reconstruct(novel_program)
            Path(exec_fname).write_text(evo_program, "utf-8")
            start_time = time.time()
            job_id = self.scheduler.submit_async(exec_fname, results_dir)
            results = self.scheduler.get_job_results(job_id, results_dir)
            rtime = time.time() - start_time

            if results.get("correct", False): 
                combined = results.get("metrics", {}).get("combined_score")
                if combined is not None:
                    if combined > ctx.deps.floor_score:
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
                            metadata={
                                "inference_time": time.time() - ctx.deps.inference_start,
                                "compute_time": rtime,
                            }
                        )
                        ray.get(self.db.add.remote(db_program))
                    else:
                        clear_results_dir(results_dir)
                        raise ModelRetry("Novel program did not achieve the minimum score on submission.")
                else:
                    clear_results_dir(results_dir)
                    raise ModelRetry("Novel program did not return a score on submission.")
            else:
                clear_results_dir(results_dir)
                raise ModelRetry("Novel program was not correct on submission. Here is the error:", results.get("error", "Unknown Error"))

        evo_explore = Agent(
            model,
            system_prompt=self.evo_config.task_sys_msg,
            deps_type=ExploreContext,
            output_type=[str, submit],
            retries=3,
            model_settings=settings)

        @evo_explore.tool(retries = 3)
        async def run_experiment(ctx: RunContext[ExploreContext], novel_program: str, change:str) -> str:
            """
            Run the novel program and confirm it is correct and check if it achieves the minimum score.

            Args:
                novel_program: A string containing a novel program that is dramatically different from the parent.
                change: A detailed description of the change made (e.g., algorithmic change,
                    refactoring, control-flow alteration, etc.).                
            Returns:
                str: feedback for the agent including correctness, score, stdout, and stderr.
            """
            if ctx.deps.run_experiment_count > 0:
                if ctx.deps.list_package_count == 0:
                    return "You must use `list_packages` to research useful packages at least once before running another experiment."                
                if ctx.deps.inspiration_count == 0:
                    return "You must use `get_inspiration` to identify useful concepts from other successful programs at least once before running another experiment."

            ctx.deps.run_experiment_count += 1

            evo_program = ctx.deps.evolve_block.reconstruct(novel_program)
            Path(exec_fname).write_text(evo_program, "utf-8")
            start_time = time.time()
            job_id = self.scheduler.submit_async(exec_fname, results_dir)
            results = self.scheduler.get_job_results(job_id, results_dir)
            rtime = time.time() - start_time

            out_str = ""
            if results.get("correct", False):
                combined = results.get("metrics", {}).get("combined_score")
                if combined is not None:
                    out_str += f"The program executed correctly.\n"
                    if combined > ctx.deps.floor_score:
                        out_str += f"It achieved a score of {combined}, which is an improvement over the minimum score of {ctx.deps.floor_score}.\n"
                    else:
                        out_str += f"It achieved a score of {combined}, which is NOT an improvement over the minimum score of {ctx.deps.floor_score}.\n"
                        out_str += "Analyze why it failed to meet the performance floor and propose a better approach.\n"
                else:
                    out_str += "Something happened and the score was not available in results. Analyze why this happened and propose a fix.\n"
            else:
                out_str += "The program did not execute correctly and did not produce a valid score. Analyze why this happened and propose a fix.\n"
        
            out_str += f"The evaluation took {rtime:.2f} seconds.\n"                
            out_str += "Here is the standard output of the program:\n"
            out_str += "```"
            out_str += results.get("stdout_log", "") + "\n"
            out_str += "```\n"
            out_str += "Here is the standard error of the program:\n"
            out_str += "```"
            out_str += results.get("stderr_log", "") + "\n"
            out_str += "```\n"

            # NOTE: This is an issue for any concurrency in this agent.
            clear_results_dir(results_dir)            
            return out_str

        @evo_explore.tool
        def get_inspiration(ctx: RunContext[ExploreContext]) -> str:
            """Call this tool to obtain concepts from other successful programs. 
               It will sample a program from the database and provide its code and its score for inspiration.
            """
            ctx.deps.inspiration_count += 1
            insp = ray.get(self.db.sample_all_topK.remote(exclude_pid=[parent.id], topK=parent_selection_top_k))
            insp_str = ""
            if not insp:
                insp_str = "No programs are available in the database for inspiration at this time."
            else:
                insp_str += f"Use this program for inspiration. It had the following score: ***{insp.combined_score}***\n"
                insp_block = extract_evolve_block(insp.code)
                insp_str += f"```python\n{insp_block.inner_content}\n```\n"
            return insp_str

        @evo_explore.tool
        def list_packages(ctx: RunContext[ExploreContext]) -> str:
            """Call this tool to list the installed packages in the execution environment."""
            ctx.deps.list_package_count += 1
            pkg_names = [name for name in sorted(dist.metadata["Name"] for dist in importlib.metadata.distributions())]
            return "Installed packages:\n" + "\n".join(pkg_names)

        @evo_explore.tool
        def install_package(ctx: RunContext[ExploreContext], package_name: str) -> str:
            """Call this tool to install a new package in the execution environment."""
            process = Popen(["uv", "pip", "install", package_name])
            process.wait()
            if process.returncode == 0:
                return f"Package {package_name} installed successfully."
            else:
                return f"Failed to install package {package_name}."

        try:
            explore_ctx = ExploreContext(evolve_block=evolve_block, floor_score=floor_score, inference_start=inference_start)
            evo_explore.run_sync(explore_prompt, deps=explore_ctx)
        except Exception as e:
            print(f"Agent encountered an error: {e}")


