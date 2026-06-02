# Run Scripts

This directory contains the experiment entrypoints.

- `run_experiments.py`: batch runner for full experiment sets
- `run_single_query.py`: one-off baseline sanity-check runner

End-to-end attack workflows are documented separately under
[../attacks/README.md](../attacks/README.md).

## Prerequisites

- activate the environment that contains the runtime dependencies
- ensure model dependencies such as `torch` are installed
- ensure the requested processed dataset and FAISS artifacts already exist under `data/`

For thesis runs, the canonical prerequisite dataset is the finalized
full-corpus close-ended set under:

- `data/processed/clean/close_ended_main/final/queries.jsonl`
- `data/processed/clean/close_ended_main/final/corpus.jsonl`

That dataset is produced through a preparation pipeline that combines:

- conservative close-ended query selection
- automated clean-run validation
- manual review to confirm answerability and acceptable answers

Attack corpora should be derived from that finalized close-ended set, and clean
comparison runs should use it directly. The reduced `close_ended_dev` dataset
remains available for faster iteration.

Typical environment on this repo:

```powershell
.\.venv\Scripts\python.exe
```

Unless noted otherwise, the `python ...` commands below assume this
virtualenv is already activated. If it is not activated, use
`.\.venv\Scripts\python.exe`.

## Defense Configuration Matrix

The canonical experiment registry is defined in `app/experiment_configs.py`.
Legacy log files may exist, but the canonical thesis configurations are the
names listed in the table below.

| Config name | Retrieval defenses | Prompt defenses | Output defenses | Thesis role / interpretation |
| --- | --- | --- | --- | --- |
| `baseline` | Standard dense top-`k` retrieval | Baseline prompt | None | Unprotected reference system |
| `retrieval_only` | `use_mmr` + `use_retrieval_filtering` | None | None | Combined retrieval-stage defense bundle |
| `prompt_only` | None | `use_prompt_isolation` + `use_instruction_hierarchy` + `use_grounding` + `use_prompt_context_flags` | None | Combined prompt-stage defense bundle |
| `output_only` | None | None | `use_output_filtering` -> `sensitive_info_filter` | Output-stage protected-value leakage filter |
| `mmr_only` | `use_mmr` only | None | None | Diversity-only retrieval defense |
| `retrieval_filtering_only` | `use_retrieval_filtering` only | None | None | Security-aware retrieval downranking only |
| `prompt_isolation_only` | None | `use_prompt_isolation` only | None | Prompt delimitation / context isolation only |
| `instruction_hierarchy_only` | None | `use_instruction_hierarchy` only | None | Explicit instruction-priority policy only |
| `grounding_only` | None | `use_grounding` only | None | Retrieved-context grounding only |
| `prompt_context_flags_only` | None | `use_prompt_context_flags` only | None | Suspicion-based prompt annotations only |

`retrieval_only` uses MMR plus security-aware retrieval filtering/downranking.
`prompt_only` enables the full prompt-defense stack. `output_only` enables the
rule-based protected-value output filter implemented in `app/output_filters.py`.

## Reproducibility Table

| Component | Canonical value | Source file / config | Reproducibility note |
| --- | --- | --- | --- |
| Clean final dataset | `close_ended_main_final` | `app/dataset_registry.py`, `scripts/data/README.md` | Canonical thesis runs should start from the finalized reviewed dataset |
| Generator profile | `main` -> `Qwen/Qwen2.5-3B-Instruct`, `4bit`, `max_new_tokens=256` | `app/model_profiles.py` | `dev` and `validation` profiles exist for iteration and confirmatory checks |
| Embedder model | `facebook/contriever` | `app/embedder.py` | Embeddings are mean-pooled and L2-normalized |
| Chunk tokenizer | `Qwen/Qwen2.5-7B-Instruct` | `scripts/indexing/indexing_presets.py` | Chunk-tokenizer changes break append-only attack-artifact compatibility |
| Chunk size / overlap | `256` / `32` | `scripts/indexing/build_retrieval_artifacts.py`, `data/built/clean/complete/retrieval_build_info.json` | Treat as structural artifact settings |
| Retrieval top-k | `5` | `app/experiment_configs.py` | Shared by all canonical configs unless code/config changes |
| Retrieval oversampling | `top_k * 4` candidate pool before MMR | `app/retrieval.py` | Applies when retrieval-stage reranking is enabled |
| Project seed | `42` by default | `app/seed_config.py` | Resolution order: explicit CLI seed, `RAG_SEED`, then default |
| Deterministic answer generation | `do_sample=False` | `app/generator.py` | Repeated clean/attack answer-generation runs are engineering checks, not independent stochastic trials |
| PoisonedRAG sampled passage generation | `do_sample=True`, `temperature=1.0` | `scripts/attacks/generate_poisoned_rag_passages.py` | Only the passage-generation stage is intentionally stochastic |
| PoisonedRAG derived sub-seeds | `derive_seed(effective_seed, query_id, passage_slot_index, slot_attempt_index)` | `app/seed_config.py`, `scripts/attacks/generate_poisoned_rag_passages.py` | This keeps passage attempts reproducible across runs with the same seed |
| Quantization | `main` and `validation` profiles use `4bit`; `dev` uses `none` | `app/model_profiles.py` | Quantized loading requires `bitsandbytes` and `accelerate` |
| CPU / CUDA behavior | `device=auto` with runtime-dependent placement | `app/model_profiles.py`, `app/generator.py`, `scripts/indexing/build_retrieval_artifacts.py` | CPU/CUDA/library differences can change latency and may prevent bitwise-identical outputs |

## Batch Experiments

Use `run_experiments.py` from the terminal.

Basic shape:

```powershell
python scripts/run/run_experiments.py --dataset-profile <preset> --config <config> --profile <profile>
```

Repeat `--config` to run multiple configurations:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile close_ended_main_final `
  --config baseline `
  --config retrieval_only `
  --config prompt_only `
  --config output_only `
  --profile main
```

You can also use:

```powershell
python scripts/run/run_experiments.py --dataset-profile close_ended_main_final --config all --profile main
```

### Common flags

- `--dataset-profile`: named dataset preset
- `--config`: experiment config name; repeat the flag or use `all`
- `--profile`: generator profile from `app/model_profiles.py`
- `--limit`: optional query cap for smoke tests only; do not use it for final thesis runs
- `--keep-existing-logs`: preserve existing per-config log files instead of replacing them

By default the runner clears the selected per-config log files before starting.
Use `--keep-existing-logs` carefully because it changes log replacement
behavior and can mix new rows with older runs.

### Available dataset presets

- `clean_complete`
- `close_ended_dev_initial`
- `close_ended_dev_final`
- `close_ended_main_initial`
- `close_ended_main_final`
- `instruction_following_close_ended_dev`
- `instruction_following_close_ended_main`
- `exfiltration_close_ended_dev`
- `exfiltration_close_ended_dev_control`
- `exfiltration_close_ended_main`
- `exfiltration_close_ended_main_control`
- `poisoned_rag_close_ended_dev`
- `poisoned_rag_close_ended_main`

### What each preset points to

- clean datasets:
  - queries under `data/processed/clean/...`
  - indexes under `data/built/clean/...`
  - logs under `logs/clean/...`
- attack datasets:
  - queries under either the finalized clean close-ended set or the attack-specific processed set
  - indexes under `data/built/attacks/...`
  - logs under `logs/attacks/...`

`close_ended_main_final` is the canonical clean experiment preset.

For the full-corpus main dataset:

- `close_ended_main_initial` reuses the `clean_complete` FAISS artifacts for
  script-validation runs
- `close_ended_main_final` reuses the same `clean_complete` FAISS artifacts for
  final clean runs

The reduced `close_ended_dev_initial` and `close_ended_dev_final` presets remain
available for iterative work.

The PoisonedRAG preset uses the finalized attack-set paths:

- queries: `data/processed/attacks/poisoned_rag/<dataset>/final/queries.jsonl`
- index: `data/built/attacks/poisoned_rag/<dataset>/faiss.index`
- metadata: `data/built/attacks/poisoned_rag/<dataset>/faiss_meta.json`
- logs: `logs/attacks/poisoned_rag/<dataset>/`

## Canonical Commands

Canonical clean run on the finalized close-ended set:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile close_ended_main_final `
  --config all `
  --profile main
```

Instruction-following attack:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile instruction_following_close_ended_main `
  --config all `
  --profile main
```

Exfiltration attack:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile exfiltration_close_ended_main `
  --config all `
  --profile main
```

Exfiltration control:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile exfiltration_close_ended_main_control `
  --config all `
  --profile main
```

PoisonedRAG attack:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile poisoned_rag_close_ended_main `
  --config all `
  --profile main
```

Smoke test with a limit:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile poisoned_rag_close_ended_main `
  --config baseline `
  --profile main `
  --limit 5
```

## Post-run Analysis

After the logs are written, use the analysis scripts to produce per-config
summaries, latency reports, and thesis-facing CSV tables.

Clean runs:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/analyze_clean_dataset.py --dataset close_ended_main --config all
```

Attack runs:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/analyze_instruction_following_attack.py --dataset close_ended_main
.\.venv\Scripts\python.exe scripts/analysis/analyze_exfiltration_attack.py --dataset close_ended_main
.\.venv\Scripts\python.exe scripts/analysis/analyze_poisoned_rag_attack.py --dataset close_ended_main
```

Combined thesis bundle:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/build_thesis_results.py --dataset close_ended_main
```

The analyzers and thesis builder are documented in
[../analysis/README.md](../analysis/README.md).

## Outputs

Each selected config writes a separate JSONL file, for example:

- `baseline_runs.jsonl`
- `retrieval_only_runs.jsonl`
- `prompt_only_runs.jsonl`
- `output_only_runs.jsonl`

These are written under the preset's log directory, for example:

- `logs/clean/close_ended_main/final/`
- `logs/attacks/instruction_following/close_ended_main/`
- `logs/attacks/exfiltration/close_ended_main/`
- `logs/attacks/poisoned_rag/close_ended_main/`

The runner also passes a combined path to the shared pipeline:

- `combined_runs.jsonl`

## Single-query Sanity Check

For a quick baseline debug pass:

```powershell
python scripts/run/run_single_query.py
```

That script accepts the same named dataset presets as the batch runner, but it
remains a debug helper rather than the canonical thesis workflow.
