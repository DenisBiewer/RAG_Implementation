# Scripts

This directory is organized by purpose:

- `data/`: dataset creation, subset selection, and finalization
- `indexing/`: chunking and FAISS index construction
- `attacks/`: attack-corpus builders
- `analysis/`: experiment-log analysis and evaluation
- `inspect/`: inspection and debugging helpers
- `run/`: experiment runners

Detailed run instructions live in [run/README.md](run/README.md).

Use the scripts from their grouped locations directly, for example:

- `python scripts/data/build_close_ended_query_set.py`
- `python scripts/indexing/build_retrieval_artifacts.py --preset close_ended_dev`
- `python scripts/run/run_experiments.py --dataset-profile close_ended_dev --config baseline --profile dev`

## PoisonedRAG workflow

The intended PoisonedRAG workflow for the `close_ended_dev` final set is staged and explicit:

1. prepare the clean final query set
2. suggest one attacker target answer per kept query
3. manually accept the suggested target answer or provide a manual replacement
4. finalize the reviewed target-answer subset
5. generate and validate poisoned passages `I`
6. build poisoned documents `P = S + I`
7. chunk and index the poisoned corpus
8. run attack experiments
9. analyze the resulting logs

The clean corpus is never modified in place. The attack corpus is written under
`data/processed/attacks/poisoned_rag/close_ended_dev/final/`.

Use the repo virtualenv for the model-backed attack scripts:

- `.\.venv\Scripts\python.exe`

### 1. Prepare the clean final set

If the final `close_ended_dev` set does not already exist, build it with the
data-preparation pipeline first. The final PoisonedRAG workflow expects:

- `data/processed/clean/close_ended_dev/final/queries.jsonl`
- `data/processed/clean/close_ended_dev/final/corpus.jsonl`

Relevant scripts in that upstream workflow include:

- `python scripts/data/build_close_ended_query_set.py`
- `python scripts/analysis/analyze_clean_baseline_validation.py`
- `python scripts/data/finalize_close_ended_query_set.py`

### 2. Suggest attacker target answers

Generate exactly one suggested fake target answer per final query:

```powershell
.\.venv\Scripts\python.exe scripts/attacks/suggest_poisoned_rag_target_answers.py
```

Optional profile override:

```powershell
.\.venv\Scripts\python.exe scripts/attacks/suggest_poisoned_rag_target_answers.py --suggester-profile main
```

Outputs:

- `data/processed/attacks/poisoned_rag/close_ended_dev/target_answers/initial/target_answer_review.jsonl`
- `data/processed/attacks/poisoned_rag/close_ended_dev/target_answers/initial/manifest.json`

Behavior:

- boolean gold answers are flipped deterministically
- non-boolean answers receive one model suggestion
- the review file must be manually approved before passage generation

### 3. Manually approve target answers

Open:

- `data/processed/attacks/poisoned_rag/close_ended_dev/target_answers/initial/target_answer_review.jsonl`

For each row, set:

- `manual_target_answer_pass`
- `manual_target_answer` when `manual_target_answer_pass = false`
- optionally `review_notes`

Semantics:

- `manual_target_answer_pass = true`
  - the generated `suggested_target_answer` is accepted automatically
- `manual_target_answer_pass = false`
  - a usable `manual_target_answer` is required and will be used instead

### 4. Finalize approved target answers

Convert the human-reviewed target-answer file into the finalized downstream
artifact:

```powershell
.\.venv\Scripts\python.exe scripts/attacks/finalize_poisoned_rag_target_answers.py
```

Outputs:

- `data/processed/attacks/poisoned_rag/close_ended_dev/target_answers/final/queries.jsonl`
- `data/processed/attacks/poisoned_rag/close_ended_dev/target_answers/final/manifest.json`

Behavior:

- requires explicit `manual_target_answer_pass = true/false` for every review row
- resolves one finalized `target_answer` per query
- uses `suggested_target_answer` when `manual_target_answer_pass = true`
- uses `manual_target_answer` when `manual_target_answer_pass = false`
- passage generation consumes the finalized file, not the raw review file

### 5. Generate and validate poisoned passages

Generate poisoned passages `I` from approved target answers and validate them
against the validator model:

```powershell
.\.venv\Scripts\python.exe scripts/attacks/generate_poisoned_rag_passages.py
```

Typical explicit call:

```powershell
.\.venv\Scripts\python.exe scripts/attacks/generate_poisoned_rag_passages.py `
  --target-answer-queries-path data/processed/attacks/poisoned_rag/close_ended_dev/target_answers/final/queries.jsonl `
  --output-passages-path data/processed/attacks/poisoned_rag/close_ended_dev/generated_passages/passages.jsonl `
  --passage-generator-profile dev `
  --validator-profile main `
  --required-passage-count 5 `
  --max-attempts-per-passage 10
```

Outputs:

- `data/processed/attacks/poisoned_rag/close_ended_dev/final/queries.jsonl`
- `data/processed/attacks/poisoned_rag/close_ended_dev/generated_passages/passages.jsonl`
- `data/processed/attacks/poisoned_rag/close_ended_dev/generated_passages/manifest.json`

Behavior:

- uses the fixed Zou-style 30-word prompt
- validates `I` with target-answer matching logic adapted from clean validation
- requires exactly `N` validated passages per query
- gives each target passage slot up to `L` attempts
- moves to the next slot immediately once one valid passage is accepted
- writes `close_ended_dev/final/queries.jsonl` with excluded queries removed
- queries that fail to fill any required slot are marked ineligible and logged with a reason

### 6. Build poisoned documents and attack corpus

Build the final poisoned documents from validated passages:

```powershell
.\.venv\Scripts\python.exe scripts/attacks/build_poisoned_rag_attack_set.py `
  --passages-path data/processed/attacks/poisoned_rag/close_ended_dev/generated_passages/passages.jsonl
```

Behavior:

- requires `--passages-path`
- uses `S = query text` verbatim
- uses `P = query.strip() + "\n\n" + I.strip()`
- writes a new attacked corpus instead of mutating the clean corpus
- builds the attack set for eligible queries only and logs dropped queries

Outputs:

- `data/processed/attacks/poisoned_rag/close_ended_dev/final/poison_docs.jsonl`
- `data/processed/attacks/poisoned_rag/close_ended_dev/final/corpus.jsonl`
- `data/processed/attacks/poisoned_rag/close_ended_dev/final/scenarios.jsonl`
- `data/processed/attacks/poisoned_rag/close_ended_dev/final/manifest.json`

### 7. Chunk and index the poisoned corpus

After building the attacked corpus, run the retrieval-artifact pipeline for the
PoisonedRAG dataset variant you use in experiments. The exact preset depends on
how your indexing presets are configured in the repo. The intended call shape is:

```powershell
python scripts/indexing/build_retrieval_artifacts.py --preset poisoned_rag_close_ended_dev
```

This step should use:

- `data/processed/attacks/poisoned_rag/close_ended_dev/final/corpus.jsonl`

### 8. Run attack experiments

Run the experiment runner against the poisoned corpus/index configuration you
prepared in the previous step:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile poisoned_rag_close_ended_dev `
  --config all `
  --profile dev
```

### 9. Analyze PoisonedRAG logs

Analyze the resulting PoisonedRAG experiment logs with:

```powershell
python scripts/analysis/analyze_poisoned_rag_attack.py --dataset close_ended_dev
```

You can repeat `--config` to restrict analysis to selected configurations:

```powershell
python scripts/analysis/analyze_poisoned_rag_attack.py `
  --dataset close_ended_dev `
  --config baseline `
  --config retrieval_only `
  --config prompt_only `
  --config output_only
```
