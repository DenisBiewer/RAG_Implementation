# PoisonedRAG

This runbook documents the staged PoisonedRAG workflow for a finalized
close-ended dataset. The canonical full-corpus workflow uses
`close_ended_main`. The reduced `close_ended_dev` variant remains available by
replacing the dataset name in the commands below.

For the default `close_ended_main` workflow, the attacked retrieval artifacts
append to the clean `clean_complete` retrieval base. For `close_ended_dev`, the
attacked retrieval artifacts append to `close_ended_dev_final`.

## Purpose

This attack builds poisoned documents `P = S + I`, where:

- `S` is the raw target query text
- `I` is a validated poisoned passage
- `P` is the indexed attack document

The clean corpus is never modified in place. The final attacked corpus is
written under `data/processed/attacks/poisoned_rag/<dataset>/final/`.

## Required Inputs

The finalized clean close-ended set must already exist:

- `data/processed/clean/close_ended_main/final/queries.jsonl`
- `data/processed/clean/close_ended_main/final/corpus.jsonl`

That finalized clean dataset is produced by the full workflow in
[../data/README.md](../data/README.md). PoisonedRAG starts from that clean
finalized set and then adds its own target-answer review, passage generation,
and passage validation stages.

## Workflow

### 1. Suggest attacker target answers

Generate exactly one suggested fake target answer per final query:

```powershell
.\.venv\Scripts\python.exe scripts/attacks/suggest_poisoned_rag_target_answers.py --dataset close_ended_main
```

Optional profile override:

```powershell
.\.venv\Scripts\python.exe scripts/attacks/suggest_poisoned_rag_target_answers.py --dataset close_ended_main --suggester-profile main
```

Outputs:

- `data/processed/attacks/poisoned_rag/close_ended_main/target_answers/initial/target_answer_review.jsonl`
- `data/processed/attacks/poisoned_rag/close_ended_main/target_answers/initial/manifest.json`

### 2. Manually approve target answers

Open:

- `data/processed/attacks/poisoned_rag/close_ended_main/target_answers/initial/target_answer_review.jsonl`

For each row, set:

- `manual_target_answer_pass`
- `manual_target_answer` when `manual_target_answer_pass = false`
- optionally `review_notes`

Semantics:

- `manual_target_answer_pass = true`
  - the generated `suggested_target_answer` is accepted automatically
- `manual_target_answer_pass = false`
  - a usable `manual_target_answer` is required and will be used instead

### 3. Finalize approved target answers

Convert the human-reviewed target-answer file into the finalized downstream
artifact:

```powershell
.\.venv\Scripts\python.exe scripts/attacks/finalize_poisoned_rag_target_answers.py --dataset close_ended_main
```

Outputs:

- `data/processed/attacks/poisoned_rag/close_ended_main/target_answers/final/queries.jsonl`
- `data/processed/attacks/poisoned_rag/close_ended_main/target_answers/final/manifest.json`

### 4. Generate and validate poisoned passages

Generate poisoned passages `I` from approved target answers and validate them
against the validator model:

```powershell
.\.venv\Scripts\python.exe scripts/attacks/generate_poisoned_rag_passages.py --dataset close_ended_main
```

Typical explicit call:

```powershell
.\.venv\Scripts\python.exe scripts/attacks/generate_poisoned_rag_passages.py `
  --dataset close_ended_main `
  --target-answer-queries-path data/processed/attacks/poisoned_rag/close_ended_main/target_answers/final/queries.jsonl `
  --output-passages-path data/processed/attacks/poisoned_rag/close_ended_main/generated_passages/passages.jsonl `
  --passage-generator-profile main `
  --validator-profile main `
  --required-passage-count 5 `
  --max-attempts-per-passage 10
```

Outputs:

- `data/processed/attacks/poisoned_rag/close_ended_main/final/queries.jsonl`
- `data/processed/attacks/poisoned_rag/close_ended_main/generated_passages/passages.jsonl`
- `data/processed/attacks/poisoned_rag/close_ended_main/generated_passages/manifest.json`

The generated-passages manifest records the passage generator profile, validator
profile, generation parameters, and excluded queries. Development-profile
artifacts should not be treated as final unless they are explicitly labeled as
such.

### 5. Build poisoned documents and attack corpus

Build the final poisoned documents from validated passages:

```powershell
.\.venv\Scripts\python.exe scripts/attacks/build_poisoned_rag_attack_set.py `
  --dataset close_ended_main `
  --passages-path data/processed/attacks/poisoned_rag/close_ended_main/generated_passages/passages.jsonl
```

Outputs:

- `data/processed/attacks/poisoned_rag/close_ended_main/final/poison_docs.jsonl`
- `data/processed/attacks/poisoned_rag/close_ended_main/final/corpus.jsonl`
- `data/processed/attacks/poisoned_rag/close_ended_main/final/scenarios.jsonl`
- `data/processed/attacks/poisoned_rag/close_ended_main/final/manifest.json`

### 6. Build retrieval artifacts

```powershell
python scripts/indexing/build_retrieval_artifacts.py --preset poisoned_rag_close_ended_main
```

This step uses:

- `data/processed/attacks/poisoned_rag/close_ended_main/final/corpus.jsonl`

Warning: `--tokenizer-name`, `--chunk-size`, `--overlap`, and the effective
input corpus/order are retrieval-artifact compatibility settings. If you
changed any of these when building the clean dataset, you must use the same
values when building attack artifacts. Otherwise, the attacked artifacts may no
longer be an append-only derivative of the clean retrieval base.

For `poisoned_rag_close_ended_main`, the clean retrieval base is
`clean_complete`. For `poisoned_rag_close_ended_dev`, the clean retrieval base
is `close_ended_dev_final`.

Output-path settings such as `--chunks-path`, `--index-path`, and `--meta-path`
change destinations only. They must point to attack-specific outputs and must
not overwrite clean artifacts.

`--device` and `--batch-size` are performance settings rather than structural
compatibility settings. They should not change the logical corpus structure,
but CPU and CUDA runs may not guarantee bit-for-bit identical embeddings.

### 7. Run experiments

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile poisoned_rag_close_ended_main `
  --config all `
  --profile main
```

### 8. Analyze logs

```powershell
python scripts/analysis/analyze_poisoned_rag_attack.py --dataset close_ended_main
```

You can repeat `--config` to restrict analysis to selected configurations:

```powershell
python scripts/analysis/analyze_poisoned_rag_attack.py `
  --dataset close_ended_main `
  --config baseline `
  --config retrieval_only `
  --config prompt_only `
  --config output_only
```

Analysis outputs:

- `results/attacks/poisoned_rag/close_ended_main/summary.json`
- `results/attacks/poisoned_rag/close_ended_main/per_query_results.jsonl`
- `results/attacks/poisoned_rag/close_ended_main/manual_review.jsonl`
- `results/attacks/poisoned_rag/close_ended_main/config_summary.csv`
- `results/attacks/poisoned_rag/close_ended_main/latency_summary.csv`

The final reported PoisonedRAG denominator may be smaller than the finalized
clean query count because only queries with approved target answers and enough
validated poisoned passages remain in the final attack set. The analyzer and
thesis tables therefore carry the explicit denominator `n` next to reported
rates.
