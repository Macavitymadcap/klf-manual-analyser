# TODO

## Core pipeline (blocking — can't run without these)
- `embedding/summarise.py` + `embedding/embed.py` — Qdrant vectorisation (optional at runtime, but the module needs to exist)
- `aggregation/aggregate.py` — cross-track statistics + recipe LLM call
- `pipeline.py` — the orchestrator that wires everything together
- `cli.py` — the typer entrypoint

## Report (substantial work)
- `report/render.py` + `report/server.py` + three Jinja2 templates + the JS audio player