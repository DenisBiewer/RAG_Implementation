# Run Scripts

This directory contains the experiment entrypoints.

- `run_experiments.py`: batch runner for full experiment sets
- `run_single_query.py`: one-off baseline sanity-check runner

## Prerequisites

- activate the environment that contains the runtime dependencies
- ensure model dependencies such as `torch` are installed
- ensure the requested processed dataset and FAISS artifacts already exist under `data/`

Typical environment on this repo:

```powershell
.\.venv\Scripts\python.exe
```

## Batch experiments

Use `run_experiments.py` from the terminal.

Basic shape:

```powershell
python scripts/run/run_experiments.py --dataset-profile <preset> --config <config> --profile <profile>
```

Repeat `--config` to run multiple configurations:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile close_ended_dev `
  --config baseline `
  --config retrieval_only `
  --config prompt_only `
  --config output_only `
  --profile dev
```

You can also use:

```powershell
python scripts/run/run_experiments.py --dataset-profile close_ended_dev --config all --profile dev
```

### Common flags

- `--dataset-profile`: named dataset preset
- `--config`: experiment config name; repeat the flag or use `all`
- `--profile`: generator profile from `app/model_profiles.py`
- `--limit`: optional query cap for smoke tests
- `--keep-existing-logs`: do not delete existing per-config log files before the run

By default the runner clears the selected per-config log files before starting.

### Available dataset presets

- `clean_dev`
- `close_ended_dev`
- `close_ended_dev_final`
- `clean_controlled_dev`
- `clean_complete`
- `instruction_following_controlled_dev`
- `exfiltration_controlled_dev`
- `exfiltration_controlled_dev_control`
- `poisoned_rag_close_ended_dev`

### What each preset points to

- clean datasets:
  - queries under `data/processed/clean/...`
  - indexes under `data/built/clean/...`
  - logs under `logs/clean/...`
- attack datasets:
  - queries under either `data/processed/clean/...` or the attack-specific processed set
  - indexes under `data/built/attacks/...`
  - logs under `logs/attacks/...`

The PoisonedRAG preset uses the finalized attack-set paths:

- queries: `data/processed/attacks/poisoned_rag/close_ended_dev/target_answers/final/queries.jsonl`
- index: `data/built/attacks/poisoned_rag/close_ended_dev/final/faiss.index`
- metadata: `data/built/attacks/poisoned_rag/close_ended_dev/final/faiss_meta.json`
- logs: `logs/attacks/poisoned_rag/close_ended_dev/`

## Canonical commands

Clean close-ended initial set:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile close_ended_dev `
  --config baseline `
  --profile dev
```

Clean close-ended final set:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile close_ended_dev_final `
  --config all `
  --profile dev
```

Instruction-following attack:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile instruction_following_controlled_dev `
  --config all `
  --profile dev
```

Exfiltration attack:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile exfiltration_controlled_dev `
  --config all `
  --profile dev
```

PoisonedRAG attack:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile poisoned_rag_close_ended_dev `
  --config all `
  --profile dev
```

Smoke test with a limit:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile poisoned_rag_close_ended_dev `
  --config baseline `
  --profile dev `
  --limit 5
```

## Outputs

Each selected config writes a separate JSONL file:

- `baseline_runs.jsonl`
- `retrieval_only_runs.jsonl`
- `prompt_only_runs.jsonl`
- `output_only_runs.jsonl`

These are written under the preset's log directory, for example:

- `logs/clean/close_ended_dev/initial/`
- `logs/clean/close_ended_dev/final/`
- `logs/attacks/instruction_following/controlled_dev/`
- `logs/attacks/exfiltration/controlled_dev/`
- `logs/attacks/poisoned_rag/close_ended_dev/`

The wrapper also passes a combined path to the shared runner:

- `combined_runs.jsonl`

## Single-query sanity check

For a quick baseline debug pass:

```powershell
python scripts/run/run_single_query.py
```

That script is still hard-wired to the clean `dev` dataset and baseline path.
