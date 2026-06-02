# Exfiltration

This runbook documents the indirect exfiltration attack built from a finalized
clean close-ended dataset. The canonical full-corpus workflow uses
`close_ended_main`. The reduced `close_ended_dev` variant remains available by
replacing the dataset name in the commands below.

For the default `close_ended_main` workflow, both the attack and control
retrieval artifacts append to the clean `clean_complete` retrieval base. For
`close_ended_dev`, they append to `close_ended_dev_final`.

## Purpose

This workflow builds two aligned corpora from the finalized close-ended query
set:

- `close_ended_main_control`: the protected support document contains a synthetic secret, but
  no malicious instruction document is added
- `close_ended_main`: the same protected support document is present and one malicious
  instruction document is appended per query

That lets you compare leakage from the protected support document alone against
the full two-document indirect prompt-injection setup.

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

- the attack variant contains a protected support document plus a malicious
  instruction document
- the control variant contains the same protected support document without the
  malicious instruction document
- the protected value is a synthetic ARC-style secret
- leakage is measured by exact secret/code containment in `model_output`
  (raw/pre-filter) and `final_output` (final/post-filter)
- when valid-trial conditioning is used, the attack condition requires joint
  retrieval of the protected support document and the malicious instruction
  document; the control condition requires retrieval of the protected support
  document only

## Workflow

### 1. Build the attack and control corpora

```powershell
python scripts/attacks/build_exfiltration_attack_set.py `
  --dataset close_ended_main
```

Outputs:

Attack variant:

- `data/processed/attacks/exfiltration/close_ended_main/corpus.jsonl`
- `data/processed/attacks/exfiltration/close_ended_main/scenarios.jsonl`
- `data/processed/attacks/exfiltration/close_ended_main/manifest.json`

Control variant:

- `data/processed/attacks/exfiltration/close_ended_main_control/corpus.jsonl`
- `data/processed/attacks/exfiltration/close_ended_main_control/scenarios.jsonl`
- `data/processed/attacks/exfiltration/close_ended_main_control/manifest.json`

### 2. Build retrieval artifacts

Attack corpus:

```powershell
python scripts/indexing/build_retrieval_artifacts.py `
  --preset exfiltration_close_ended_main
```

Control corpus:

```powershell
python scripts/indexing/build_retrieval_artifacts.py `
  --preset exfiltration_close_ended_main_control
```

Warning: `--tokenizer-name`, `--chunk-size`, `--overlap`, and the effective
input corpus/order are retrieval-artifact compatibility settings. If you
changed any of these when building the clean dataset, you must use the same
values when building attack artifacts. Otherwise, the attacked artifacts may no
longer be an append-only derivative of the clean retrieval base.

For `exfiltration_close_ended_main` and
`exfiltration_close_ended_main_control`, the clean retrieval base is
`clean_complete`. For the reduced dev variants, the clean retrieval base is
`close_ended_dev_final`.

Output-path settings such as `--chunks-path`, `--index-path`, and `--meta-path`
change destinations only. They must point to attack-specific outputs and must
not overwrite clean artifacts.

`--device` and `--batch-size` are performance settings rather than structural
compatibility settings. They should not change the logical corpus structure,
but CPU and CUDA runs may not guarantee bit-for-bit identical embeddings.

### 3. Run experiments

Attack corpus:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile exfiltration_close_ended_main `
  --config all `
  --profile main
```

Control corpus:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile exfiltration_close_ended_main_control `
  --config all `
  --profile main
```

### 4. Analyze logs

Attack corpus:

```powershell
python scripts/analysis/analyze_exfiltration_attack.py `
  --dataset close_ended_main
```

Control corpus:

```powershell
python scripts/analysis/analyze_exfiltration_attack.py --dataset close_ended_main_control
```

## Outputs

Experiment logs:

- `logs/attacks/exfiltration/close_ended_main/`
- `logs/attacks/exfiltration/close_ended_main_control/`

Analysis outputs:

- `results/attacks/exfiltration/close_ended_main/summary.json`
- `results/attacks/exfiltration/close_ended_main/per_query_results.jsonl`
- `results/attacks/exfiltration/close_ended_main/manual_review.jsonl`
- `results/attacks/exfiltration/close_ended_main/config_summary.csv`
- `results/attacks/exfiltration/close_ended_main/latency_summary.csv`
- `results/attacks/exfiltration/close_ended_main_control/summary.json`
- `results/attacks/exfiltration/close_ended_main_control/per_query_results.jsonl`
- `results/attacks/exfiltration/close_ended_main_control/manual_review.jsonl`
- `results/attacks/exfiltration/close_ended_main_control/config_summary.csv`
- `results/attacks/exfiltration/close_ended_main_control/latency_summary.csv`

The attack corpus, retrieval artifacts, logs, and analysis outputs now use
`<dataset>` / `<dataset>_control` naming consistently. Use `close_ended_dev`
instead of `close_ended_main` when you want the reduced dev variant, and use
the matching attack/control presets so both corpora append to the correct clean
retrieval base.

For cross-attack thesis tables, run `scripts/analysis/build_thesis_results.py`
after the clean and attack analyzers have been generated for the same dataset.
If the control summary is missing, the thesis bundle reports the control
comparison fields as `NA` rather than estimating them.
