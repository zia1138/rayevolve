# rayevolve <a href="https://github.com/zia1138/rayevolve"><img src="rayevolve.png" align="right" height="150" alt="rayevolve github" /></a>

Experimental project for LLM guided algorithm design and evolution built on [ray](https://www.ray.io/),
[pydantic-ai](https://github.com/pydantic/pydantic-ai), and [logfire](https://github.com/pydantic/logfire).
Originally started as an in-place fork from [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve).

# Installation

Clone the repo and setup a `uv` environment as follows:
```bash
git clone git@github.com:zia1138/rayevolve.git
cd rayevolve
uv sync
source .venv/bin/activate
```

Create an account and project called `rayevolve` on [logfire](https://github.com/pydantic/logfire). 
You will need to authenticate to logfire and select the `rayevolve` project to log your runs. You can do this as follows:
```bash
logfire auth
logfire projects use rayevolve
```

# Quickstart Instructions

You can run the circle packing example using the configs in `examples/circlepacking/config.py` with the `default` profile as follows:
```bash
rayevolve run examples/circle_packing
```
Use `rayevolve --help` to get all of the command line parameters. 

You can visualize the progress as follows:
`python -m rayevolve.webui.visualization --db results_XXX_YYY/evolution_db.sqlite`

Output program data into a parquet file:
`python -m rayevolve.utils.load_df results_XXX_YYY/evolution_db.sqlite -o test.parquet`

We use a a config-as-code system where you use can initialzie data classes in your projects `config.py` to modify any parameters. See the 
file [src/rayevolve/core/common.py](src/rayevolve/core/common.py)
all the configuration parameters along with a function to validate these
parameters. 

# TODO

## Priority
- [ ] continue to clean up config system
- [ ] add multiple model support (try to get to 2.635 on circle packing)

## Important but not urgent
- [ ] integrate and experiment with examples from openevolve use to drive/priortize improvement
- [ ] handle runtime timeouts correctly (took too long to run)
- [ ] identify codebaes and projects with good evaluators + benchmarks to migrate into tool
- [ ] handle logging correctly on ray cluster (get logs to driver somehow)

## Algorithmic Ideas
- [ ] dynamically optimizing and updating the system prompt based on where the system is getting stuck
- [ ] add a meta-optimizer to get the system unstuck and adds strategic guidance 
- [ ] for exploit/explore improve probe functionality, not clear it is always useful

# Related Open Source Projects

* [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)
* [LLM4AD](https://github.com/Optima-CityU/llm4ad)
* [GigaEvo](https://github.com/AIRI-Institute/gigaevo-platform/tree/main)
* [station](https://github.com/dualverse-ai/station)
* [CSE/EvoControl](https://github.com/QuantaAlpha/EvoControl)

# rayevolve on Lightning AI Studios

We provide a script [mmt_cluster.sh](mmt_cluster.sh) to launch a ray cluster on Lightning AI Studios using 
multimachine training (MMT).

Install Lightning AI Studios SDK within the uv environment:
```bash
cd rayevolve && uv sync
source .venv/bin/activate
uv pip install lightning-sdk
```
Authenticate logfire and select rayevolve project:
```bash
logfire auth
logfire projects use rayevolve
```
Then launch an MMT cluster using the following command:
```bash
lightning run mmt --command="/teamspace/studios/this_studio/rayevolve/mmt_cluster.sh"
```
Run it on the ray cluster using 
```bash
rayevolve run examples/circle_packing --ray-ip x.x.x.x
```
where x.x.x.x is the IP address of the ray cluster head node, which you can find from the log output
on Lightning AI Studios.

# debugging in ray

If the ray VSCode debugger doesn't work.
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

