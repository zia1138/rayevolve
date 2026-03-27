from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

import textwrap

SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in combinatorial optimization, specifically the Flexible Job Shop Scheduling Problem (FJSP).

    Problem Description:
    The Flexible Job Shop Scheduling Problem (FJSP) aims to assign operations of jobs to compatible machines and determine their processing sequence to minimize the makespan (total completion time). Given a set of jobs, each consisting of a sequence of operations, and a set of machines, where each operation can be processed on one or more machines with potentially different processing times, the objective is to:
    1. Assign each operation to exactly one compatible machine
    2. Determine the processing sequence of operations on each machine
    3. Minimize the makespan (completion time of the last operation)

    The problem has the following constraints:
    - Each operation must be processed on exactly one machine from its set of compatible machines
    - Operations of the same job must be processed in their predefined order (precedence constraints)
    - Each machine can process only one operation at a time
    - No preemption is allowed (once an operation starts, it must finish without interruption)
    - All jobs are available at time zero

    Task:
    - Implement the `solve` function that assigns operations to machines and determines start times.
    - Minimize the makespan (completion time of the last operation).
    - All precedence and machine capacity constraints must be satisfied.
    - If any constraint is violated, the solution receives no score.

    Input args:
        instance_id   : (str) Unique identifier for this problem instance.
        num_jobs      : (int) Number of jobs.
        num_machines  : (int) Number of machines.
        jobs          : (list) A list of jobs, where each job is a list of operations.
                        Each operation is a list of (machine, time) pairs representing
                        compatible machines and their processing times. Items are 1-indexed.

    Returns:
        A dict with keys:
            - "makespan" (float): The completion time of the last operation.
            - "machine_assignments" (list): Machine assigned to each operation (globally indexed).
            - "start_times" (list): Start time of each operation (globally indexed).

    Key insights to explore:
    1. Greedy dispatching rules (shortest processing time, earliest due date)
    2. Priority-based scheduling with machine load balancing
    3. Local search and neighborhood structures
    4. Metaheuristics (genetic algorithms, simulated annealing, tabu search)
    5. Constraint propagation to tighten time windows
    6. Decomposition into assignment and sequencing subproblems

    IMPORTANT: The main entry point is `def solve(instance_id, num_jobs, num_machines, jobs)`.
""")


def list_profiles() -> list[str]:
    """List available configuration profiles to display on CLI."""
    return ["default"]


def build_strategy_model() -> ModelSpec:
    return ModelSpec(
        description="GEMINI 3 Flash Preview",
        model=GoogleModel("gemini-3-flash-preview"),
        settings=GoogleModelSettings(),
    )


def build_evo_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            description="GEMINI 3 Flash Preview",
            model=GoogleModel("gemini-3-flash-preview"),
            settings=GoogleModelSettings(google_thinking_config={"thinking_budget": 8192})
        )
    ]


def get_config(profile: str = "default") -> RayEvolveConfig:
    """Get configuration for the given profile."""
    if profile == "default":
        return RayEvolveConfig(
            evo=EvolutionConfig(task_sys_msg=SYSTEM_MSG,
                                build_strategy_model=build_strategy_model,
                                build_evo_models=build_evo_models,
                                num_agent_workers=1),
            backend=BackendConfig(),
        )
    raise ValueError(f"Unknown profile: {profile}")
