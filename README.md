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

# TODO

- [ ] integrate and experiment with examples from openevolve use to drive/priortize improvement
- [ ] pilot code translation idea and PyO3
- [ ] add multiple model support (try to get to 2.635 on circle packing)
- [ ] remove more unused ShinkaEvolve code

# Ideas

Need to motivate/prioritize these based on solving problems and benchmarking.
- dynamically optimizing and updating the system prompt based on where the system is getting stuck
- add a metaoptimizer to get the system unstuck and adds strategic guidance 
- for exploit/explore improve probe functionality, not clear it is useful
- handle runtime timeouts correctly (took too long to run)

## Higher Priority

- identify other problems where ShinkaEvolve will perform poorly
- identify codebaes and projects with good evaluators + benchmarks to migrate into tool

## Lower Priority

- migrate codebase to another language orig_code output == opt_code output (in say pyO3?)
    - generate tests in original language
    - compare input and output equivalence in second language, focus on use of cffi
    - after requivalence optimize and innovate (speed up)
    - use codex or codex-like agent
    - basis for full codebase optimization/innovation
- migrate pytorch code base to improved and faster data loader    
- make a custom compression algorithm validator is decompress(compression(orig)) == orig
- select an ML problem or area (e.g. flowmatching/text diffusion) with good benchmarks to apply approaches
- use proof validator for validation in rayevolve lean, coq, sympy 
- optimize SQL queries validator is orig query result = optimized query result + speed

# Using rayevolve to improve tabular DL models

## TabArena Setup Instructions

Sets up [TabArena](https://github.com/autogluon/tabrepo) locally in the 
`./test` directory. Framework seems heavy and might not be suitible for rayevolve.

```bash
mkdir test
cd test
uv init -p 3.11 
uv sync
source .venv/bin/activate
git clone https://github.com/autogluon/autogluon.git
./autogluon/full_install.sh 
git clone git@github.com:autogluon/tabrepo.git
uv pip install --prerelease=allow -e ./tabrepo/tabarena[benchmark]
cd tabrepo/examples/benchmarking 
python run_quickstart_tabarena.py 
cd custom_tabarena_model
python run_custom_model_on_tabarena_lite.py 
python run 
```


## Protein Ideas
- https://graphein.ai/ (graph rep for proteins, etc.)
- https://github.com/mims-harvard/TDC/blob/main/tutorials/graphein_demo_developability.ipynb (graph rep for proteins)
- https://tdcommons.ai/single_pred_tasks/develop/ (TDC task)
