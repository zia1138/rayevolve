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
    parent_code: str = Field(description="Code of the parent program being improved.")
    parent_score: float = Field(description="Score of the parent program being improved.")


class DriftContext(BaseModel):
    parent_code: str = Field(description="Code of the parent program being modified.")

class NovelProgram(BaseModel):
    """Use this when you have discovered a novel and correct program.
       Your novel program must include any unchanged code above "EVOLVE-BLOCK-START" and below the 
       line containing "EVOLVE-BLOCK-END". 
    """
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



#from __future__ import annotations

import difflib
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EvolveErrorType(str, Enum):
    TAGS_NOT_PRESENT = "tags_not_present"
    MULTIPLE_START_TAGS = "multiple_start_tags"
    MULTIPLE_END_TAGS = "multiple_end_tags"
    START_AFTER_END = "start_after_end"

    CHANGED_ABOVE = "changed_above"
    CHANGED_BELOW = "changed_below"
    NOT_CHANGED_BETWEEN = "not_changed_between"


@dataclass(frozen=True)
class TagPositions:
    start_idx: Optional[int] = None  # 0-based line index
    end_idx: Optional[int] = None    # 0-based line index


@dataclass(frozen=True)
class EvolveVerificationResult:
    ok: bool
    errors: tuple[tuple[EvolveErrorType, str], ...]
    original_tags: TagPositions
    updated_tags: TagPositions
    above_changed: Optional[bool] = None
    between_changed: Optional[bool] = None
    below_changed: Optional[bool] = None


def verify_evolve_block(
    original_text: str,
    updated_text: str,
    start_tag: str = "EVOLVE-BLOCK-START",
    end_tag: str = "EVOLVE-BLOCK-END",
    *,
    require_tags_on_own_line: bool = False,
) -> EvolveVerificationResult:
    def split_lines(s: str) -> list[str]:
        return s.splitlines()

    def find_tag_lines(lines: list[str], tag: str) -> list[int]:
        if require_tags_on_own_line:
            return [i for i, ln in enumerate(lines) if ln.strip() == tag]
        return [i for i, ln in enumerate(lines) if tag in ln]

    def get_positions(lines: list[str]) -> TagPositions:
        starts = find_tag_lines(lines, start_tag)
        ends = find_tag_lines(lines, end_tag)
        return TagPositions(
            start_idx=starts[0] if starts else None,
            end_idx=ends[0] if ends else None,
        )

    orig_lines = split_lines(original_text)
    upd_lines = split_lines(updated_text)

    o_starts = find_tag_lines(orig_lines, start_tag)
    o_ends = find_tag_lines(orig_lines, end_tag)
    u_starts = find_tag_lines(upd_lines, start_tag)
    u_ends = find_tag_lines(upd_lines, end_tag)

    errors: list[tuple[EvolveErrorType, str]] = []

    def validate(which: str, starts: list[int], ends: list[int]) -> None:
        if not starts or not ends:
            errors.append((EvolveErrorType.TAGS_NOT_PRESENT, f"{which}: missing start and/or end tag"))
            return
        if len(starts) > 1:
            errors.append((EvolveErrorType.MULTIPLE_START_TAGS, f"{which}: found {len(starts)} start tags"))
        if len(ends) > 1:
            errors.append((EvolveErrorType.MULTIPLE_END_TAGS, f"{which}: found {len(ends)} end tags"))
        if len(starts) == 1 and len(ends) == 1 and starts[0] > ends[0]:
            errors.append((EvolveErrorType.START_AFTER_END, f"{which}: start tag occurs after end tag"))

    validate("original", o_starts, o_ends)
    validate("updated", u_starts, u_ends)

    original_tags = get_positions(orig_lines)
    updated_tags = get_positions(upd_lines)

    fatal = {
        EvolveErrorType.TAGS_NOT_PRESENT,
        EvolveErrorType.MULTIPLE_START_TAGS,
        EvolveErrorType.MULTIPLE_END_TAGS,
        EvolveErrorType.START_AFTER_END,
    }
    if any(et in fatal for et, _ in errors):
        return EvolveVerificationResult(
            ok=False,
            errors=tuple(errors),
            original_tags=original_tags,
            updated_tags=updated_tags,
            above_changed=None,
            between_changed=None,
            below_changed=None,
        )

    # Single, ordered tags in both documents
    o_start, o_end = o_starts[0], o_ends[0]
    u_start, u_end = u_starts[0], u_ends[0]

    o_above = orig_lines[:o_start]
    o_between = orig_lines[o_start + 1 : o_end]
    o_below = orig_lines[o_end + 1 :]

    u_above = upd_lines[:u_start]
    u_between = upd_lines[u_start + 1 : u_end]
    u_below = upd_lines[u_end + 1 :]

    above_same = (o_above == u_above)
    below_same = (o_below == u_below)
    between_changed = (o_between != u_between)

    if not above_same:
        errors.append((EvolveErrorType.CHANGED_ABOVE, "Content changed above the start tag line"))
    if not below_same:
        errors.append((EvolveErrorType.CHANGED_BELOW, "Content changed below the end tag line"))
    if not between_changed:
        errors.append((EvolveErrorType.NOT_CHANGED_BETWEEN, "No changes detected between tag lines"))

    return EvolveVerificationResult(
        ok=(len(errors) == 0),
        errors=tuple(errors),
        original_tags=TagPositions(start_idx=o_start, end_idx=o_end),
        updated_tags=TagPositions(start_idx=u_start, end_idx=u_end),
        above_changed=(not above_same),
        between_changed=between_changed,
        below_changed=(not below_same),
    )


def llm_fix_instructions(
    original_text: str,
    updated_text: str,
    result: EvolveVerificationResult,
    *,
    start_tag: str = "EVOLVE-BLOCK-START",
    end_tag: str = "EVOLVE-BLOCK-END",
    require_tags_on_own_line: bool = False,
    max_diff_lines: int = 400,
) -> str:
    """
    Returns plain instructions for an LLM on how to fix updated_text to satisfy rules.
    IMPORTANT: This function ONLY shows diffs for *disallowed* changes (ABOVE/BELOW).
               It never shows diffs of allowed BETWEEN edits.
    """

    def split_lines(s: str) -> list[str]:
        return s.splitlines()

    def unified(a_lines: list[str], b_lines: list[str], fromfile: str, tofile: str) -> str:
        diff_iter = difflib.unified_diff(a_lines, b_lines, fromfile=fromfile, tofile=tofile, lineterm="")
        diff_lines = list(diff_iter)
        if not diff_lines:
            return "(no diff)"
        if len(diff_lines) > max_diff_lines:
            head = diff_lines[: max_diff_lines // 2]
            tail = diff_lines[-max_diff_lines // 2 :]
            return "\n".join(head + ["... (diff truncated) ..."] + tail)
        return "\n".join(diff_lines)

    def has_error(t: EvolveErrorType) -> bool:
        return any(code == t for code, _ in result.errors)

    def fmt_line(idx0: Optional[int]) -> str:
        return "N/A" if idx0 is None else str(idx0 + 1)

    # If OK, minimal instruction.
    if result.ok:
        return (
            "All constraints are satisfied.\n"
            "- Tags are present.\n"
            "- Content outside the tags is unchanged.\n"
            "- Content between the tags has changed.\n"
        )

    orig_lines = split_lines(original_text)
    upd_lines = split_lines(updated_text)

    lines: list[str] = []
    lines.append("Fix the updated code to satisfy these constraints:")
    lines.append(f"- There must be exactly one '{start_tag}' line and exactly one '{end_tag}' line.")
    lines.append("- Do not change any content above the start tag line.")
    lines.append("- Do not change any content below the end tag line.")
    lines.append("- Make at least one change strictly between the tag lines.")
    lines.append("")
    lines.append("Tag locations (1-based line numbers):")
    lines.append(f"- original: start={fmt_line(result.original_tags.start_idx)}, end={fmt_line(result.original_tags.end_idx)}")
    lines.append(f"- updated:  start={fmt_line(result.updated_tags.start_idx)}, end={fmt_line(result.updated_tags.end_idx)}")
    lines.append("")

    # Tag / structure errors: explicit corrective steps (no diffs; regions not reliable).
    fatal = {
        EvolveErrorType.TAGS_NOT_PRESENT,
        EvolveErrorType.MULTIPLE_START_TAGS,
        EvolveErrorType.MULTIPLE_END_TAGS,
        EvolveErrorType.START_AFTER_END,
    }
    if any(code in fatal for code, _ in result.errors):
        if has_error(EvolveErrorType.TAGS_NOT_PRESENT):
            lines.append("ERROR: Tags are missing.")
            lines.append("Fix:")
            lines.append(f"- Add exactly one line containing '{start_tag}' and exactly one line containing '{end_tag}'.")
            lines.append("- Place them so the editable block is between them.")
            lines.append("- Keep everything outside the tags identical to the original.")
            lines.append("")
        if has_error(EvolveErrorType.MULTIPLE_START_TAGS):
            lines.append("ERROR: Multiple start tags found.")
            lines.append("Fix:")
            lines.append(f"- Remove extra '{start_tag}' lines so only one remains.")
            lines.append("")
        if has_error(EvolveErrorType.MULTIPLE_END_TAGS):
            lines.append("ERROR: Multiple end tags found.")
            lines.append("Fix:")
            lines.append(f"- Remove extra '{end_tag}' lines so only one remains.")
            lines.append("")
        if has_error(EvolveErrorType.START_AFTER_END):
            lines.append("ERROR: Start tag occurs after end tag.")
            lines.append("Fix:")
            lines.append(f"- Reorder so '{start_tag}' appears before '{end_tag}'.")
            lines.append("")
        lines.append("After fixing tags, make edits ONLY between the tags.")
        return "\n".join(lines)

    # Safe to segment.
    o_start = result.original_tags.start_idx
    o_end = result.original_tags.end_idx
    u_start = result.updated_tags.start_idx
    u_end = result.updated_tags.end_idx
    assert None not in (o_start, o_end, u_start, u_end)

    o_above = orig_lines[:o_start]
    u_above = upd_lines[:u_start]
    o_between = orig_lines[o_start + 1 : o_end]
    u_between = upd_lines[u_start + 1 : u_end]
    o_below = orig_lines[o_end + 1 :]
    u_below = upd_lines[u_end + 1 :]

    if has_error(EvolveErrorType.CHANGED_ABOVE):
        lines.append("ERROR: Disallowed changes detected ABOVE the start tag.")
        lines.append("Fix:")
        lines.append("- Revert the ABOVE region to exactly match the original.")
        lines.append("Diff to undo (original -> updated) for ABOVE:")
        lines.append("```diff")
        lines.append(unified(o_above, u_above, "original:ABOVE", "updated:ABOVE"))
        lines.append("```")
        lines.append("")

    if has_error(EvolveErrorType.CHANGED_BELOW):
        lines.append("ERROR: Disallowed changes detected BELOW the end tag.")
        lines.append("Fix:")
        lines.append("- Revert the BELOW region to exactly match the original.")
        lines.append("Diff to undo (original -> updated) for BELOW:")
        lines.append("```diff")
        lines.append(unified(o_below, u_below, "original:BELOW", "updated:BELOW"))
        lines.append("```")
        lines.append("")

    if has_error(EvolveErrorType.NOT_CHANGED_BETWEEN):
        lines.append("ERROR: No changes detected BETWEEN the tags (this region must change).")
        lines.append("Fix:")
        lines.append("- Make at least one edit strictly between the tag lines.")
        lines.append("- Do not change the tag lines.")
        lines.append("- Do not change anything above or below the tags.")
        lines.append("")

        # Optional: if BETWEEN is empty, suggest adding something.
        if len(o_between) == 0 and len(u_between) == 0:
            lines.append("Note: The region between the tags is currently empty; add content there.")
            lines.append("")

    lines.append("Stop when:")
    lines.append("- ABOVE and BELOW match the original exactly,")
    lines.append("- and BETWEEN differs from the original.")
    return "\n".join(lines)

import re
from typing import Optional

_FENCE_RE = re.compile(r"^\s*```[^\n]*\n", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\n\s*```\s*$", re.IGNORECASE)

def strip_outer_blank_lines(text: str) -> str:
    """
    Remove ALL leading and trailing blank lines (including whitespace-only lines).
    Does not touch internal blank lines.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1

    end = len(lines)
    while end > start and not lines[end - 1].strip():
        end -= 1

    return "\n".join(lines[start:end])


def unwrap_markdown_code_fences(text: str) -> str:
    """
    If the content is wrapped in a single outer Markdown fence, remove it.
    If multiple fenced blocks exist, return the largest fenced block.
    """
    # Fast path: entire content is a single fenced block
    if _FENCE_RE.search(text) and _FENCE_END_RE.search(text):
        text2 = _FENCE_RE.sub("", text, count=1)
        text2 = _FENCE_END_RE.sub("", text2, count=1)
        return text2

    # Otherwise, extract fenced blocks and return the largest
    blocks = []
    for m in re.finditer(r"```[^\n]*\n([\s\S]*?)\n```", text):
        blocks.append(m.group(1))
    if blocks:
        return max(blocks, key=len)

    return text


def normalize_llm_program(text: str) -> str:
    """
    Normalization order:
      1) Normalize line endings
      2) Remove leading/trailing blank lines
      3) Remove outer Markdown code fences
      4) Remove leading/trailing blank lines again (fences often introduce them)

    Result is stable for diffing.
    """
    # 1) Normalize line endings
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2) Strip blank lines first (prevents fake ABOVE/BELOW diffs)
    t = strip_outer_blank_lines(t)

    # 3) Remove Markdown fences
    t = unwrap_markdown_code_fences(t)

    # 4) Strip blank lines again (post-fence cleanup)
    t = strip_outer_blank_lines(t)

    return t


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
            if random.random() < 0.5:
                self.agent_driftaway(current_gen)
            else:
                if random.random() < 0.5:
                    self.agent_climb(current_gen)
                else:
                    self.agent_climb_or_drift(current_gen, drift_up=True)

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
            # Sample from non-elite programs for drift up.
            elite_parent = ray.get(self.db.sample_all_programs.remote())
        else:        
            elite_parent = ray.get(self.db.sample_archive_program.remote())

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
            - You must `run_experiment` before `log_experiment` for each hypothesis.                             
            - **Safety:** 
              - You may only modify code that lies below a line containing "EVOLVE-BLOCK-START" 
                and above a line containing "EVOLVE-BLOCK-END". 
                You must NOT remove or modify any code outside these tags.
                You must NOT remove the tags themselves.                         
              - Make sure your rewritten program maintains the same inputs and outputs as the original program, 
                but with a novel internal implementation.
              - Make sure the file still runs after your changes.                        

            ### COMPLETION
            - If you achieve `new_score > {score}`, submit your improved program using `submit_tool`.
            - If you cannot beat the score after 5 distinct hypotheses, give up and offer an explanation why.
         """)

        coder_prompt = coder_template.format(lang=self.evo_config.language, 
                                             code=elite_parent.code, 
                                             score=elite_parent.combined_score)

        model = GoogleModel('gemini-2.5-flash')
        settings = GoogleModelSettings(google_thinking_config={"thinking_budget":-1})

        def submit_tool(ctx: RunContext[ClimbContext], program: str) -> None:
            """Call this tool to submit your improved program when you have achieved a higher score."""
            program = normalize_llm_program(program)
            res = verify_evolve_block(ctx.deps.parent_code, program)
            if not res.ok:
                raise ModelRetry(llm_fix_instructions(ctx.deps.parent_code, program, res))
            
            Path(exec_fname).write_text(program, "utf-8")
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
                            code=program,
                            language=self.evo_config.language,
                            parent_id=elite_parent.id,
                            generation=current_gen,
                            code_diff="agent_climb",
                            embedding=[],
                            correct=True,
                            combined_score=combined,
                        )
                        ray.get(self.db.add.remote(db_program))
                    else:
                        clear_results_dir(results_dir)
                        raise ModelRetry("Improved program did not achieve a higher score upon re-evaluation.")
                else:
                    clear_results_dir(results_dir)
                    raise ModelRetry("Improved program did not return a score upon re-evaluation.")
            else:
                clear_results_dir(results_dir)
                raise ModelRetry("Improved program was not correct upon re-evaluation.")
            
        evo_coder = Agent(
            model,
            system_prompt=self.evo_config.task_sys_msg,
            deps_type=ClimbContext,
            output_type=[str, submit_tool],
            retries=3,            
            model_settings=settings)

        @evo_coder.tool(retries = 3)
        def run_experiment(ctx: RunContext[ClimbContext], program: str, hypothesis: str) -> str:
            """Call this tool with a novel program that you want to evaluate. It will return
            the results of executing the program including its score and correctness.
            Provide a string with the hypothesis you are testing with this program.
            """

            # If code outside of the evo block is changed, raise ModelRetry
            program = normalize_llm_program(program)
            res = verify_evolve_block(ctx.deps.parent_code, program)
            if not res.ok:
                raise ModelRetry(llm_fix_instructions(ctx.deps.parent_code, program, res))
            
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

            # NOTE: This is an issue for any concurrency in this agent.
            clear_results_dir(results_dir)
            return out_str


        #debugpy.listen(5678)
        #debugpy.wait_for_client()
        #debugpy.breakpoint()        
        try:
            evo_coder.run_sync(coder_prompt, 
                               deps=ClimbContext(parent_code=normalize_llm_program(elite_parent.code), 
                                                 parent_score=elite_parent.combined_score))
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
            ### MISSION
            The code is below. Your goal is to produce a different solution that still works correctly.   
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
            - **Safety:** 
              - You may only modify code that lies below a line containing "EVOLVE-BLOCK-START" 
                and above a line containing "EVOLVE-BLOCK-END". 
                You must NOT remove or modify any code outside these tags.
                You must NOT remove the tags themselves.                         
              - Make sure your rewritten program maintains the same inputs and outputs as the original program, 
                but with a novel internal implementation.
              - Make sure the file still runs after your changes.
                                                      
            ### COMPLETION
            - If change type was verified and the program is substantially different from the parent, use the submit_novel tool to submit your novel program.
            - If you cannot find a correct and novel program after 5 attempts explain why and give up.
         """)

        coder_prompt = coder_template.format(lang=self.evo_config.language, 
                                             code=elite_parent.code)

        model = GoogleModel('gemini-2.5-flash')
        settings = GoogleModelSettings(google_thinking_config={"thinking_budget":-1})

        def submit_novel(ctx: RunContext[ClimbContext], program: str) -> None:
            """Call this tool to submit your novel program."""
            program = normalize_llm_program(program)
            res = verify_evolve_block(ctx.deps.parent_code, program)
            if not res.ok:
                raise ModelRetry(llm_fix_instructions(ctx.deps.parent_code, program, res))
            
            Path(exec_fname).write_text(program, "utf-8")
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
                        code=program,
                        language=self.evo_config.language,
                        parent_id=elite_parent.id,
                        generation=current_gen,
                        code_diff="agent_climb",
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
            deps_type=DriftContext,
            output_type=[str, submit_novel],
            retries=3,
            model_settings=settings)

        evo_diff = Agent(model, output_type=VerifiedChangeAndNovel | NotNovel | ChangeNotVerified,
                         model_settings=settings)

        async def confirm_change(parent_code: str, novel_code: str, change_type: str) -> str:
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
            r = await evo_diff.run(diff_prompt)
            return r.output             

        @evo_coder.tool(retries = 3)
        async def check_correctness(ctx: RunContext[DriftContext], program: str, change_type:str) -> str:
            """Call this tool with a novel program that you want to check for correctness and a description
            of the change type you made. It will return the results of executing the program including its correctness. 
            It will check that the change type you provided was applied and if the program is substantially different from the parent.
            """

            program = normalize_llm_program(program)
            res = verify_evolve_block(ctx.deps.parent_code, program)
            if not res.ok:
                raise ModelRetry(llm_fix_instructions(ctx.deps.parent_code, program, res))
            #debugpy.listen(5678)
            #debugpy.wait_for_client()
            #debugpy.breakpoint()   

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
                    confirmation = await confirm_change(ctx.deps.parent_code, program, change_type)
                    if isinstance(confirmation, VerifiedChangeAndNovel):
                        out_str += "The change type was verified and the program is substantially different from the parent.\n"
                    elif isinstance(confirmation, ChangeNotVerified):
                        out_str += "The change type could not be verified.\n"
                    elif isinstance(confirmation, NotNovel):
                        out_str += "Change type was verified, but the program is not substantially different from the parent.\n"
                else:
                    out_str += "The program did not execute correctly.\n"
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
            agent_result = evo_coder.run_sync(coder_prompt, deps=DriftContext(parent_code=normalize_llm_program(elite_parent.code)))
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
