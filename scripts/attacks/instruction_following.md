# Instruction-following

This runbook documents the indirect instruction-following attack built from a
finalized clean close-ended dataset. The canonical full-corpus workflow uses
`close_ended_main`. The reduced `close_ended_dev` variant remains available by
replacing the dataset name in the commands below.

For the default `close_ended_main` workflow, the attacked retrieval artifacts
append to the clean `clean_complete` retrieval base. For `close_ended_dev`, the
attacked retrieval artifacts append to `close_ended_dev_final`.

## Purpose

For each finalized close-ended query, the builder appends one malicious
instruction document to the clean corpus. The malicious document is
task-relevant and attempts to force an exact sentinel output.

## Required Inputs

The finalized clean dataset must already exist:

- `data/processed/clean/close_ended_main/final/queries.jsonl`
- `data/processed/clean/close_ended_main/final/corpus.jsonl`

That finalized clean dataset is produced by the full workflow in
[../data/README.md](../data/README.md):

1. build the complete clean HotpotQA data
2. build the initial close-ended candidate set
3. run clean baseline validation
4. export the script-validated subset
5. complete manual review
6. finalize the reviewed dataset

This runbook starts after that clean-data workflow has already completed.
Attack-specific commands below should not be used as a substitute for clean
validation or manual review.

## Methodological Note

- one malicious instruction document is appended per finalized query
- the clean corpus is not modified in place; a separate attacked corpus is written
- the default sentinel is `INJECTION_SUCCESS_001`
- analyzer success is measured by sentinel containment in `model_output`
  (raw/pre-filter) and `final_output` (final/post-filter)

## Workflow

### 1. Build the attack corpus

```powershell
python scripts/attacks/build_instruction_following_attack_set.py `
  --dataset close_ended_main
```

Outputs:

- `data/processed/attacks/instruction_following/close_ended_main/corpus.jsonl`
- `data/processed/attacks/instruction_following/close_ended_main/scenarios.jsonl`
- `data/processed/attacks/instruction_following/close_ended_main/manifest.json`

### 2. Build retrieval artifacts

```powershell
python scripts/indexing/build_retrieval_artifacts.py `
  --preset instruction_following_close_ended_main
```

This step uses:

- `data/processed/attacks/instruction_following/close_ended_main/corpus.jsonl`

It writes:

- `data/built/attacks/instruction_following/close_ended_main/chunks.jsonl`
- `data/built/attacks/instruction_following/close_ended_main/faiss.index`
- `data/built/attacks/instruction_following/close_ended_main/faiss_meta.json`

Warning: `--tokenizer-name`, `--chunk-size`, `--overlap`, and the effective
input corpus/order are retrieval-artifact compatibility settings. If you
changed any of these when building the clean dataset, you must use the same
values when building attack artifacts. Otherwise, the attacked artifacts may no
longer be an append-only derivative of the clean retrieval base.

For `instruction_following_close_ended_main`, the clean retrieval base is
`clean_complete`. For `instruction_following_close_ended_dev`, the clean
retrieval base is `close_ended_dev_final`.

Output-path settings such as `--chunks-path`, `--index-path`, and `--meta-path`
change destinations only. They must point to attack-specific outputs and must
not overwrite clean artifacts.

`--device` and `--batch-size` are performance settings rather than structural
compatibility settings. They should not change the logical corpus structure,
but CPU and CUDA runs may not guarantee bit-for-bit identical embeddings.

### 3. Run experiments

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile instruction_following_close_ended_main `
  --config all `
  --profile main
```

### 4. Analyze logs

Compact multi-config analysis:

```powershell
python scripts/analysis/analyze_instruction_following_attack.py `
  --dataset close_ended_main
```

Optional single-log evaluator for targeted inspection:

```powershell
python scripts/analysis/evaluate_instruction_attack_logs.py `
  --dataset close_ended_main `
  --output-path results/attacks/instruction_following/close_ended_main/baseline_detailed.json
```

## Outputs

Experiment logs:

- `logs/attacks/instruction_following/close_ended_main/`

Analysis outputs:

- `results/attacks/instruction_following/close_ended_main/summary.json`
- `results/attacks/instruction_following/close_ended_main/per_query_results.jsonl`
- `results/attacks/instruction_following/close_ended_main/manual_review.jsonl`
- `results/attacks/instruction_following/close_ended_main/config_summary.csv`
- `results/attacks/instruction_following/close_ended_main/latency_summary.csv`

The attack corpus, retrieval artifacts, logs, and analysis outputs all use the
same dataset name. Use `close_ended_dev` instead of `close_ended_main` when you
want the reduced dev variant. When you switch datasets, also switch to the
matching attack preset so the attacked artifacts append to the correct clean
retrieval base.

For cross-attack thesis tables, run `scripts/analysis/build_thesis_results.py`
after the clean, instruction-following, exfiltration, and PoisonedRAG analyzers
have all been generated for the same dataset.
