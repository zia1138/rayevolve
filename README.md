# rayevolve <a href="https://github.com/zia1138/rayevolve"><img src="rayevolve.png" align="right" height="150" alt="rayevolve github" /></a>
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

# TODO: V1 MVP

✦ Yes, you have nailed it. This is a complete, logically sound, and robust "V1" Adaptive Evolutionary System.

  You have successfully distilled the complexity into a clean State -> Strategy -> Action loop.

  Here is the final confirmation of your logic to ensure we are 100% aligned before you build it:

  1. The Signal (The Strategist's Eyes)
   * Data: A plot of Best Score vs Time.
   * Logic:s
       * Steep Slope: "We are winning." -> CLIMB.
       * Shallow Slope: "We are slowing down." -> DRIFT UP.
       * Flat (Short): "We are stuck locally." -> DRIFT AWAY.
       * Flat (Long): "We are dead." -> JUMP.

  2. The Actions (The Worker's Instructions)


  ┌────────────┬───────────────────────────────┬──────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────┐
  │ Mode       │ Parent Selection              │ Goal / Acceptance Criteria               │ Why?                                                                                    │
  ├────────────┼───────────────────────────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ CLIMB      │ Elites Only (Archive)         │ Strict Improvement (child > parent)      │ Maximize the current peak. Don't dilute the pool.                                       │
  │ DRIFT UP   │ All Programs (Biased to Good) │ Strict Improvement (child > parent)      │ Find new peaks near the current ones.                                                   │
  │ DRIFT AWAY │ All Programs (Biased to Good) │ Correctness Only (child.correct == True) │ Walk sideways across the landscape. Accumulate neutral mutations to find a new path up. │
  │ JUMP       │ None (Use Archive as Context) │ Novelty / Difference                     │ Teleport to a new region. Break structural fixations.                                   │
  └────────────┴───────────────────────────────┴──────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────┘


  3. The Implementation (Simplicity)
   * Islands: You are starting with 1 Island. This removes the complexity of migration and comparative benchmarking, letting you focus entirely on tuning the Time/Score dynamics.
   * Worker: The worker is just a "machine" that takes the Mode and executes the loop. It handles the specific prompting and retry logic (the "Fixer") internally.

  Verdict:
  This is the correct architecture. It is simple enough to debug but powerful enough to exhibit complex adaptive behavior. Build this.

  ┌────────────┬────────────────────────────────────────────────────────┬────────────────────────────────────┐
  │ Mode       │ Prompt Intent                                          │ Acceptance Criteria                │
  ├────────────┼────────────────────────────────────────────────────────┼────────────────────────────────────┤
  │ CLIMB      │ "Optimize this specific logic."                        │ score > parent                     │
  │ DRIFT UP   │ "Improve this general approach."                       │ score > parent                     │
  │ DRIFT AWAY │ "Keep functionality, change implementation/structure." │ correct == True AND code != parent │
  │ JUMP       │ "Ignore previous code. Propose a novel solution."      │ correct == True                    │
  └────────────┴────────────────────────────────────────────────────────┴────────────────────────────────────┘


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





