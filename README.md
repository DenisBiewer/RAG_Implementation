# RAG Implementation

This repository contains the thesis code for dataset construction, retrieval
artifact generation, experiment execution, and attack analysis for the RAG
pipeline.

## Thesis Overview

This repository supports a BSc thesis that evaluates lightweight defenses for
RAG security in a controlled local pipeline. The evaluated attack families are:

1. instruction-following indirect prompt injection
2. exfiltration of a synthetic protected ARC-style value
3. PoisonedRAG-style retrieval poisoning

The implemented defense stages are:

1. retrieval stage
2. prompt-construction stage
3. output-processing stage

The canonical final dataset is `close_ended_main`. The reduced
`close_ended_dev` dataset exists for faster iteration and debugging only. The
primary experiment outputs are JSONL logs in `logs/`, analyzer summaries in
`results/`, and thesis-facing tables in `results/thesis/<dataset>/`.

If you need the intended workflow from raw HotpotQA through finalized
close-ended datasets and then into the attack-specific pipelines, start with
[scripts/README.md](scripts/README.md) and
[scripts/data/README.md](scripts/data/README.md). Retrieval-artifact build
details and append-only attack-index compatibility rules are documented in
[scripts/indexing/README.md](scripts/indexing/README.md).

## Repo Layout

- `app/`: pipeline, models, configs, and shared runtime code
- `scripts/`: operational scripts for data prep, indexing, experiments, and analysis
- `data/`: processed corpora, query sets, and built retrieval artifacts
- `logs/`: experiment outputs written by the runners
- `results/`: downstream analysis outputs
- `tests/`: test coverage

## Scope

This repository evaluates controlled local RAG attack/defense experiments. It
does not evaluate:

- ingestion-time defenses
- model fine-tuning
- adversarial retriever training
- external tools or function calling
- multi-user settings or conversational memory

The experiments should therefore not be read as a complete security evaluation
of all deployed RAG systems. Final thesis conclusions should be drawn from the
logged final runs and their analyzer outputs, not from legacy or development
logs.

## RAG Architecture Summary

Pipeline:

`HotpotQA-derived corpus -> chunking + metadata -> Contriever embeddings -> FAISS retrieval -> prompt construction -> Qwen generation -> output processing/filtering -> JSONL logging`

| Stage | Implementation summary |
| --- | --- |
| HotpotQA-derived corpus | `scripts/data/build_hotpot_data.py` builds a plain-text clean corpus from HotpotQA `fullwiki` train and validation contexts and a query pool from the validation split. |
| Chunking + metadata | `scripts/indexing/build_retrieval_artifacts.py` chunks documents with tokenizer `Qwen/Qwen2.5-7B-Instruct`, chunk size `256`, overlap `32`, and keeps document/attack metadata with each chunk. |
| Contriever embeddings | `app/embedder.py` loads `facebook/contriever`, mean-pools token states over non-padding tokens, and L2-normalizes embeddings. |
| FAISS retrieval | `app/vector_index.py` uses exact `faiss.IndexFlatIP`; because embeddings are normalized, inner product is cosine-equivalent. Canonical runs use `top_k = 5` from `app/experiment_configs.py`. |
| Prompt construction | `app/prompt_builder.py` builds a two-message chat prompt from the query and retrieved chunks, with optional prompt defenses for isolation, instruction hierarchy, grounding, and context flags. |
| Qwen generation | `app/model_profiles.py` defines the canonical `main` profile as `Qwen/Qwen2.5-3B-Instruct` with `4bit` quantization. `app/generator.py` uses deterministic answer generation for standard experiments with `do_sample=False`, and the batch runners can opt into sampled generation with `--temperature <value>`. |
| Output processing/filtering | `app/output_filters.py` applies the rule-based `sensitive_info_filter` when output filtering is enabled. |
| JSONL logging | `app/pipeline.py` and `app/experiment_runner.py` write one JSONL record per query/configuration with retrieved chunks, prompt, raw output, final output, support coverage, and timing fields. |

## Canonical Thesis Artifacts

`close_ended_main` is the canonical thesis dataset. `close_ended_dev` is for
iteration only.

| Artifact type | Canonical path | Purpose | Notes |
| --- | --- | --- | --- |
| Clean final queries and corpus | `data/processed/clean/close_ended_main/final/` | Canonical clean dataset used by final clean runs and as the starting point for all attack families | `close_ended_dev` mirrors the same structure for iteration only |
| Clean complete retrieval artifacts | `data/built/clean/complete/` | Canonical retrieval base for `close_ended_main` clean and attack workflows | Build settings are recorded in `retrieval_build_info.json` |
| Instruction-following attack corpus, scenarios, manifest | `data/processed/attacks/instruction_following/close_ended_main/` | Appended malicious instruction documents plus evaluation metadata | Built from the finalized clean set; one malicious document per query |
| Exfiltration attack corpus, scenarios, manifest | `data/processed/attacks/exfiltration/close_ended_main/` | Attack variant with protected support documents plus malicious instruction documents | Use together with the matched control variant |
| Exfiltration control corpus, scenarios, manifest | `data/processed/attacks/exfiltration/close_ended_main_control/` | Control variant with protected support documents but no malicious instruction documents | Used to compare leakage without the indirect instruction document |
| PoisonedRAG generated passages and manifest | `data/processed/attacks/poisoned_rag/close_ended_main/generated_passages/` | Records approved/validated poisoned passages plus generator and validator metadata | Development-profile passage artifacts should not be treated as final unless explicitly labeled |
| PoisonedRAG final queries, poison docs, corpus, scenarios, manifest | `data/processed/attacks/poisoned_rag/close_ended_main/final/` | Final retrieval-poisoning attack artifacts | Canonical denominator can be smaller than the clean query count |
| Final logs | `logs/clean/close_ended_main/final/` and `logs/attacks/*/close_ended_main*/` | Per-config experiment logs | Temperature-enabled variants use nested subdirectories such as `temperature_0_7/`; treat legacy/dev logs as non-canonical unless explicitly cited |
| Analyzer outputs | `results/validation/close_ended_main/` and `results/attacks/*/close_ended_main*/` | Per-attack and clean summaries, per-query outputs, latency tables | Temperature-enabled variants mirror the same nested subdirectory convention as the logs |
| Thesis tables | `results/thesis/close_ended_main/` | Cross-attack CSV tables and manifest used for thesis-facing reporting | Generated by `scripts/analysis/build_thesis_results.py` |

Note: The tables in the written thesis are manually formatted in LaTeX for readability and page layout. Their values were transcribed from the canonical analyzer outputs and thesis-facing CSV/JSON summaries under `results/`. The scripts generate the source result summaries, but not necessarily the exact LaTeX code used in the submitted thesis.

## Environment Setup

The checked local virtual environment in `.venv/pyvenv.cfg` targets Python
`3.11.9`.

Create and activate a virtual environment in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Unless noted otherwise, the `python ...` commands in this repository assume
that this virtual environment is already activated. If it is not activated, use
`.\.venv\Scripts\python.exe` explicitly.

Setup notes:

- `requirements.txt` is a compatibility/dependency file, not a full lockfile.
- Retrieval and small non-quantized runs can execute on CPU, but the canonical
  `main` and `validation` generator profiles use `4bit` quantization and are
  CUDA-oriented in practice because `app/generator.py` relies on
  `bitsandbytes` and `accelerate` for quantized loading.
- First runs download Hugging Face models and tokenizers for Contriever and the
  selected Qwen profile. These artifacts are expected to be cached locally by
  the Hugging Face stack.
- Quantized profile loading requires `bitsandbytes`. If quantized loading is
  unavailable, use a non-quantized profile or verify the local CUDA stack.
- CPU/CUDA differences, library versions, and backend kernels may change
  latencies and can prevent bitwise-identical results even when seeds match.
- In the checked local environment, targeted repository tests run with
  `unittest`; `pytest` is not installed in `.venv`.

## Canonical Thesis Workflow

Use the canonical final dataset and follow the high-level chain below:

1. prepare the clean dataset through the full workflow in
   [scripts/data/README.md](scripts/data/README.md)
2. build or verify the compatible retrieval artifacts for the selected stage
   from [scripts/indexing/README.md](scripts/indexing/README.md)
3. run the clean final experiments from
   [scripts/run/README.md](scripts/run/README.md)
4. build the attack-specific corpora from `scripts/attacks/`
5. run the attack experiments
6. run the analyzers from [scripts/analysis/README.md](scripts/analysis/README.md)
7. build the thesis tables with `scripts/analysis/build_thesis_results.py`

Attack runbooks start only after the finalized clean dataset already exists.

## Running Experiments

The canonical thesis experiment basis is the finalized `close_ended_main`
dataset. It uses the complete clean corpus and a reviewed close-ended query set
selected through the same initial-selection, script-validation, and manual-review
workflow used for the development dataset.

Typical clean baseline run:

```powershell
python scripts/run/run_experiments.py --dataset-profile close_ended_main_final --config baseline --profile main
```

Temperature-enabled clean baseline run:

```powershell
python scripts/run/run_experiments.py --dataset-profile close_ended_main_final --config baseline --profile main --temperature 0.7
```

Detailed experiment-running instructions, dataset presets, canonical defense
configurations, and reproducibility notes are documented in
[scripts/run/README.md](scripts/run/README.md).

## Results Analysis

After runs complete, use the analysis scripts to produce thesis-facing summaries
from the current `close_ended_main` logs.

Per-analysis commands:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/analyze_clean_dataset.py --dataset close_ended_main --config all
.\.venv\Scripts\python.exe scripts/analysis/analyze_instruction_following_attack.py --dataset close_ended_main
.\.venv\Scripts\python.exe scripts/analysis/analyze_exfiltration_attack.py --dataset close_ended_main
.\.venv\Scripts\python.exe scripts/analysis/analyze_poisoned_rag_attack.py --dataset close_ended_main
```

Temperature-enabled analysis uses the same commands with a matching temperature:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/analyze_clean_dataset.py --dataset close_ended_main --config all --temperature 0.7
.\.venv\Scripts\python.exe scripts/analysis/analyze_instruction_following_attack.py --dataset close_ended_main --temperature 0.7
.\.venv\Scripts\python.exe scripts/analysis/analyze_exfiltration_attack.py --dataset close_ended_main --temperature 0.7
.\.venv\Scripts\python.exe scripts/analysis/analyze_poisoned_rag_attack.py --dataset close_ended_main --temperature 0.7
```

Combined thesis tables:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/build_thesis_results.py --dataset close_ended_main
```

The analyzers emit:

- `summary.json`
- `per_query_results.jsonl`
- `manual_review.jsonl`
- `config_summary.csv`
- `latency_summary.csv`

For temperature-enabled runs, the JSONL logs and analyzer outputs are written to
matching nested subdirectories such as:

- `logs/clean/close_ended_main/final/temperature_0_7/`
- `results/validation/close_ended_main/temperature_0_7/`

The thesis builder writes compact cross-attack tables under
`results/thesis/<dataset>/`. Source-level metric definitions, denominator
rules, and log-field notes are documented in
[scripts/analysis/README.md](scripts/analysis/README.md).

## Run Log Timing Fields

Experiment run logs record both stage timing and direct defense timing:

- `latencies_ms`: end-to-end stage timing for the executed query path, such as
  embedding, retrieval, prompt building, generation, and output filtering
- `defense_latencies_ms`: direct defense-code timing only, measured around the
  concrete defense functions and prompt-construction branches

Prompt-stage defenses can also affect generation latency indirectly by changing
prompt length. That indirect cost is reflected in `latencies_ms`, not in
`defense_latencies_ms`.

Token-count fields are also logged per run:

- `prompt_tokens`: total token count of the final chat prompt sent to the model
- `output_tokens`: token count of the raw model output before output filtering
- `retrieved_context_tokens`: token count of the retrieved-context section that
  was inserted into the prompt

## Script Overview

The broader script layout is documented in [scripts/README.md](scripts/README.md).

## Attribution and References

This project uses the [HotpotQA dataset](https://hotpotqa.github.io/) (Yang et al., 2018), distributed under the [Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) license. The original dataset was processed to construct the retrieval corpus and evaluation query sets used in this research. Redistributed HotpotQA-derived dataset material remains subject to the applicable CC BY-SA 4.0 requirements.

**HotpotQA reference:** Yang et al. (2018). [*HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*](https://aclanthology.org/D18-1259/). Proceedings of EMNLP 2018.

Academic references for the other datasets, models, and methods used in this project are provided in the accompanying bachelor thesis, which is also available in this repository.
