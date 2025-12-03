# rayevolve
Experimental project for LLM guided algorithm design and evolution built on ray.
Based on [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve).


You can run the circle packing example as follows:
`python -m rayevolve.launch_hydra variant@_global_=circle_packing_example`

You can visualize the progress as follows:
`python -m rayevolve.webui.visualization --db results/rayevolve_circle_packing/[current run]/evolution_db.sqlite`

# debugging in ray

Add the following in `~/.vscode/launch.json`. You can also use in the launch.json file
the "Add Configuration..." button in the lower right. Then chose "Python Debugger" and
"Remote Attach" Use local host and the matching port number for the code below. I think you 
might need different ports for different actors.
```json
{
  "configurations": [
    {
      "name": "Python Debugger: Remote Attach",
      "type": "debugpy",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "."
        }
      ]
    }
  ]
}
```

In the ray worker/actor add
```python
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
debugpy.breakpoint()
```
Then hit the green arrow in the debug panel to attach and debug.

This is inspired from https://github.com/ray-project/ray/issues/41953
and https://code.visualstudio.com/docs/python/debugging#_debugging-by-attaching-over-a-network-connection

# TODO

EvoStrategist
  * Role: The high-level planner for the entire evolutionary process.
  * Mechanism: A single LLM call that performs a holistic analysis of the evolutionary state.
  * Input: A comprehensive "Strategic Context Dossier" (including historical performance, archive diversity, island reports, and top program examples).
  * Output: A structured JSON directive (e.g., "EXPLOIT" or "EXPLORE") that guides the EvoCoder.
  * Purpose: To make critical, data-driven judgments to avoid stagnation and optimize the search trajectory.

EvoCoder

  * Role: The iterative developer responsible for executing the EvoStrategist's directives.
  * Mechanism: A multi-turn agent loop: generate code → evaluate → reflect → refine → repeat until successful.
  * Input: A strategic directive from the EvoStrategist.
  * Output: A validated, improved Program object (or a notification of failed attempts).
  * Purpose: To reliably produce high-quality, evolved programs through an iterative development and debugging process.
EvoCoder's Key Tools
  1. update_plan(plan: List[Dict]): For maintaining its internal, iterative work plan.
  2. sample_program(strategy: str, metric: Optional[str] = None, ...): To intelligently select a parent program and relevant inspirations.
  3. evaluate_program(code_string: str, parent_id: str): To run generated code and receive detailed performance/error feedback.
  4. submit_program(program_object: Program, parent_id: str): To add a successfully developed program to the database.


Here is a consolidated list of all the fields needed to provide the LLM with a comprehensive view of the evolutionary state, enabling it to make informed decisions about exploitation and exploration, and diagnose various failure modes:

    1 {
    2   "island_id": 1,
    3   "current_generation": 50,
    4   
    5   "score_history": [
    6     // Captures: Stagnation (Plateau Detection)
    7     // Tracks the maximum combined_score for this island at significant generation milestones.
    8     {"gen": 40, "best_score": 0.85},
    9     {"gen": 45, "best_score": 0.88},
   10     {"gen": 50, "best_score": 0.88} 
   11   ],
   12 
   13   "recent_island_activity_stats": {
   14     // Captures: Viability Collapse (The Syntax Wall), overall recent success rate.
   15     // Summarizes recent activity for the entire island.
   16     "total_attempts_last_10_gens": 50,  // Total programs submitted by this island in last 10 generations.
   17     "valid_children_rate_last_10_gens": 0.05, // % of submitted programs that were 'correct'.
   18     "most_common_failure_reason": "SyntaxError: unmatched parenthesis", // Provides diagnostic hint.
   19     "avg_code_complexity_last_10_gens": 120 // Trend for Code Bloat (if scores are flat but complexity increases).
   20   },
   21 
   22   "archive": [
   23     // Captures: Lineage Collapse, Code Bloat, Overfitting, Operational Bloat, Evaluation Flakiness.
   24     // This is a snapshot of the elite programs for this specific island,
   25     // providing detailed quality and effort metrics per program.
   26     {
   27       "id": "prog_102",
   28       "parent_id": "prog_99", 
   29       "code_snippet": "def solve_problem(input_data):\n    # Complex but effective solution\n    ... [truncated for brevity, LLM reads full relevant code] ...", 
   30       
   31       "public_score": 0.95,       // Primary fitness metric.
   32       "private_score": 0.40,      // (CRITICAL) Detects Overfitting if significantly lower than public_score.
   33       "runtime_sec": 12.5,        // (CRITICAL) Detects Operational Bloat if high compared to baseline.
   34       "child_avg_score": 0.20,    // (CRITICAL) Detects Evaluation Flakiness if significantly lower than parent's public_score.
   35       
   36       "effort_stats": {
   37         // Captures: Local Optima (The Wall), Neglect vs. Stubbornness.
   38         "times_selected_as_parent": 20,       // How many times this program was picked as a parent.
   39         "avg_turns_per_successful_attempt": 4.5, // Average LLM turns needed to generate a valid child.
   40         "improvement_rate_from_this_parent": "0%", // % of times a child from this parent beat its score.
   41         "most_common_child_failure": "TimeoutError" // Specific issue children from this parent face.
   42       }
   43     },
   44     // ... more archive programs for this island ...
   45   ]
   46 }

  This comprehensive payload allows the LLM to:

   * Diagnose Stagnation: By observing score_history over time.
   * Identify Lineage Collapse: By checking parent_id within the archive.
   * Detect Code Bloat: By analyzing code_snippet content and avg_code_complexity_last_10_gens.
   * Spot Viability Collapse: Through valid_children_rate_last_10_gens and most_common_failure_reason.
   * Uncover Overfitting: By comparing public_score and private_score.
   * Address Operational Bloat: By examining runtime_sec.
   * Recognize Evaluation Flakiness: By comparing public_score with child_avg_score.
   * Distinguish "Stuck" from "Neglected": Using effort_stats.


  Here is how the data maps to specific worker configurations:

1. The "Deep Optimization" Worker (High Exploitation)
  * Trigger:
      * score_history is rising (positive velocity).
      * valid_children_rate is high (code is healthy).
      * effort_stats shows low turn counts (easy gains).
  * Worker Configuration:
      * Task: Improve the current best program.
      * Strategy: diff (small patches).
      * Turns: Low (1-2 turns). Don't waste tokens; it's easy.
      * Temperature: Low (0.2). Precision over creativity.
      * Parent Selection: power_law (Best).

2. The "Wall Breaker" Worker (Targeted Exploration)
  * Trigger:
      * score_history is flat (plateau).
      * effort_stats shows high times_selected with 0% improvement (Stubborn Parent).
      * code_snippet looks reasonable (not bloated).
  * Worker Configuration:
      * Task: Break the local optimum for this specific parent.
      * Strategy: multi_turn (Chain of Thought).
      * Turns: High (5-10 turns). Give the LLM space to "think" and self-correct.
      * Temperature: High (0.8). Force diverse ideas.
      * Prompt: "This code is stuck. Identify the bottleneck and propose a radically different approach."

3. The "Refactoring" Worker (Maintenance)
  * Trigger:
      * valid_children_rate is crashing (Syntax Wall).
      * avg_code_complexity is spiking (Bloat).
      * runtime_sec is high (Operational Bloat).
  * Worker Configuration:
      * Task: Simplify and clean the code without changing logic.
      * Strategy: full (Rewrite).
      * Goal: Maximize correctness and speed, ignore score gain.
      * Prompt: "Refactor this code to be more readable and efficient. Do not add features."

4. The "Colonizer" Worker (Radical Exploration)
  * Trigger:
      * island_lag is high (this island is failing compared to neighbors).
      * archive shows Lineage Collapse (echo chamber).
  * Worker Configuration:
      * Task: Inject new genetic material.
      * Strategy: cross (Crossover) or Migration.
      * Input: Take a high-scoring parent from a different island.
      * Prompt: "Hybridize this local program with this high-scoring visitor from Island B."

Summary

You are moving from a stochastic system (randomly rolling dice for explore/exploit) to a cybernetic system (feedback loops driving intelligent allocation).

  * Data: The fields we listed (Archive, History, Effort, Yield).
  * Decision: The Supervisor LLM analyzes the data.
  * Action: It outputs a JSON config for the next EvoWorker job (e.g., {"mode": "wall_breaker", "target_parent": "prog_102", "turns": 8}).

