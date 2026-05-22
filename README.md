# RAG Implementation

This repository contains the thesis code for dataset construction, retrieval
artifact generation, experiment execution, and attack analysis for the RAG
pipeline.

## Repo layout

- `app/`: pipeline, models, configs, and shared runtime code
- `scripts/`: operational scripts for data prep, indexing, experiments, and analysis
- `data/`: processed corpora, query sets, and built retrieval artifacts
- `logs/`: experiment outputs written by the runners
- `results/`: downstream analysis outputs
- `tests/`: test coverage

## Running experiments

The canonical experiment entrypoint is:

```powershell
python scripts/run/run_experiments.py --dataset-profile close_ended_dev --config baseline --profile dev
```

Detailed experiment-running instructions, dataset presets, and common commands
are documented in [scripts/run/README.md](scripts/run/README.md).

## Script overview

The broader script layout is documented in [scripts/README.md](scripts/README.md).
