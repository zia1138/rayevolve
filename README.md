# rayevolve
Experimental project for LLM guided algorithm design and evolution built on ray.
Based on [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve).


You can run the circle packing example as follows:
`python -m rayevolve.launch_hydra variant@_global_=circle_packing_example`

You can visualize the progress as follows:
`python -m rayevolve.webui.visualization --db results/rayevolve_circle_packing/[current run]/evolution_db.sqlite`


# TODO

- need to correctly handle generation counter
- add improved mata-analyzer
