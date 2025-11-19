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

Evals
- figure out how to get to circle packing 2.6x with fewer models and fewer generation
- better restart (exploration) when system gets stuck
- add exploration prompts versus exploitation prompts, new ideas
- add multiturn agents to evolution system
- get other examples from openevolve into framework
- explore use of shinkaevolve for a verifable comp bio algorithm with good evals to compare against
 
Ideas
- simplify the codebase
- get meta summarizer working again
- switch to codex (headless) or pydantic-ai for evolution
- incorporate more feedback into new program construction
- switch to modal, etc for program eval (if not local)
- add an agent that monitors the simulation status and dynaimcally adjust exploration/exploitation
