"""Rayevolve config for FrontierCO Capacitated Facility Location Problem.

Adaptation differences from the canonical FrontierCO evaluation
(https://huggingface.co/datasets/CO-Bench/FrontierCO/blob/main/CFLP/config.py):

  1. solve() returns a single dict instead of yielding progressively better
     solutions. In FrontierCO, solvers are generators with a 10s per-instance
     wall-clock budget; the last yielded solution before timeout is scored.
     Here, evolution provides iteration, so solve() returns once.

  2. Timeout is enforced as a total budget across all parallel Ray tasks
     (default 60s via ray.wait), not a per-instance 10s wall-clock limit.
     Since all dev instances run concurrently, each effectively gets the
     full timeout window.

  3. Normalization uses the same primal-gap formula as FrontierCO:
         1 - |score - optimal| / max(score, optimal)
     This differs from the CO-Bench aircraft_landing example which uses
     optimal / score.

  4. FrontierCO's load_data has a bug ("n": m instead of "n": n). We fix
     this. All benchmark instances have n == m so behaviour is identical.

  5. solve() receives an extra kwarg `instance_id` (str) not present in the
     FrontierCO interface, for logging consistency with other rayevolve examples.
"""

import textwrap
from datetime import datetime
from pathlib import Path

from rayevolve.core.common import RayEvolveConfig, EvolutionConfig, BackendConfig, ModelSpec
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings


def list_profiles() -> list[str]:
    return ["default", "test", "prod"]


SYSTEM_MSG = textwrap.dedent("""\
    You are an expert in combinatorial optimization, specifically the Capacitated Facility Location Problem (CFLP).

    Problem Description:
    Given a set of n potential facility locations and m customers, the goal is to decide which
    facilities to open and how to assign customer demands to open facilities so as to minimize
    the total cost. The total cost consists of fixed costs for opening facilities plus
    transportation costs for serving customer demands.

    Mathematical formulation:
    - Decision variables:
        facilities_open[i] in {0, 1}  for i = 0..n-1  (1 if facility i is open)
        assignments[j][i] >= 0        for j = 0..m-1, i = 0..n-1  (demand of customer j served by facility i)

    - Minimize:
        sum(fixed_cost[i] * facilities_open[i] for i in range(n))
      + sum((assignments[j][i] / demands[j]) * trans_costs[i][j] for j in range(m) for i in range(n))

    - Subject to:
        1. Demand satisfaction: for each customer j,
           sum(assignments[j][i] for i in range(n)) == demands[j]
        2. Facility dependency: assignments[j][i] == 0 if facilities_open[i] == 0
        3. Capacity constraints: for each facility i,
           sum(assignments[j][i] for j in range(m)) <= capacities[i]
        4. Non-negativity: assignments[j][i] >= 0

    IMPORTANT: trans_costs[i][j] represents the cost of allocating customer j's ENTIRE demand
    from facility i. For a partial allocation of `a` units (out of demands[j] total), the cost
    is (a / demands[j]) * trans_costs[i][j]. This is equivalent to a per-unit cost of
    trans_costs[i][j] / demands[j].

    Task:
    - Implement the `solve` function that decides which facilities to open and how to allocate
      customer demands to minimize total cost.
    - If any constraint is violated, the solution receives no score.

    Input kwargs:
        instance_id  : (str) Unique identifier for this problem instance.
        n            : (int) Number of potential facilities.
        m            : (int) Number of customers.
        capacities   : (list of float) Capacity of each facility, length n.
        fixed_cost   : (list of float) Fixed cost for opening each facility, length n.
        demands      : (list of float) Demand of each customer, length m.
        trans_costs  : (list of lists) trans_costs[i][j] is the cost of serving customer j's
                       entire demand from facility i. Shape: n x m.

    Returns:
        A dict with keys:
            "total_cost"       : (float) The computed total cost.
            "facilities_open"  : (list of int) Binary list of length n (1=open, 0=closed).
            "assignments"      : (list of lists) assignments[j][i] is the amount of customer j's
                                 demand served by facility i. Shape: m x n.

    Time budget:
    The FrontierCO benchmark allows 10 seconds per instance. Your solution must complete
    within this budget. Avoid algorithms with worst-case exponential runtime on 100x100
    instances. Prefer efficient heuristics over exact methods.

    Key insights to explore:
    1. Greedy: open cheapest facilities, assign customers to nearest open facility
    2. LP relaxation followed by rounding
    3. Lagrangian relaxation on demand or capacity constraints
    4. Local search: swap open/closed facilities, reassign customers
    5. Simulated annealing or tabu search on facility open/close decisions
    6. Once facilities are chosen, assignment is a min-cost transportation problem
    7. For each fixed facility set, optimal assignment can be found greedily or via LP
    8. Instance sizes range from 100x100 (dev) to 2000x2000 (hard test)

    IMPORTANT: The main entry point is `def solve(**kwargs)`.
""")


def build_strategy_model() -> ModelSpec:
    return ModelSpec(
        description="GEMINI 3 Flash Preview",
        model=GoogleModel("gemini-3-flash-preview"),
        settings=GoogleModelSettings(),
    )


def build_evo_models() -> list[ModelSpec]:
    gemini = ModelSpec(
        description="Gemini 3 Flash Preview (thinking)",
        model=GoogleModel("gemini-3-flash-preview"),
        settings=GoogleModelSettings(google_thinking_config={"thinking_budget": 8192}),
    )
    gemini_pro = ModelSpec(
        description="Gemini 3 Pro Preview (thinking)",
        model=GoogleModel("gemini-3-pro-preview"),
        settings=GoogleModelSettings(google_thinking_config={"thinking_budget": 16384}),
    )
    # Gemini Flash 80%, Gemini Pro 20%
    return [gemini] * 4 + [gemini_pro] * 1


RAYEVOLVE_ROOT = Path(__file__).resolve().parent.parent.parent  # rayevolve/
RESULTS_DIR = RAYEVOLVE_ROOT / "results" / "FrontierCO__CFLP"


def _make_results_dir(prefix: str = "") -> str:
    """Create a timestamped results dir under rayevolve/results/FrontierCO__CFLP/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{timestamp}" if prefix else f"results_{timestamp}"
    return str(RESULTS_DIR / name)


def get_config(profile: str = "default") -> RayEvolveConfig:
    if profile == "default":
        return RayEvolveConfig(
            evo=EvolutionConfig(
                task_sys_msg=SYSTEM_MSG,
                build_strategy_model=build_strategy_model,
                build_evo_models=build_evo_models,
                results_dir=_make_results_dir("default"),
            ),
            backend=BackendConfig(),
        )
    if profile == "test":
        return RayEvolveConfig(
            evo=EvolutionConfig(
                task_sys_msg=SYSTEM_MSG,
                build_strategy_model=build_strategy_model,
                build_evo_models=build_evo_models,
                max_generations=6,
                num_agent_workers=6,
                results_dir=_make_results_dir("test"),
            ),
            backend=BackendConfig(),
        )
    if profile == "prod":
        return RayEvolveConfig(
            evo=EvolutionConfig(
                task_sys_msg=SYSTEM_MSG,
                build_strategy_model=build_strategy_model,
                build_evo_models=build_evo_models,
                max_generations=60,
                num_agent_workers=12,
                results_dir=_make_results_dir("prod"),
            ),
            backend=BackendConfig(),
        )
    raise ValueError(f"Unknown profile: {profile}")
