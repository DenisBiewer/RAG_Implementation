# Attack Workflows

This directory contains the attack-corpus builders plus the workflow runbooks
for each attack family.

Unless noted otherwise, the `python ...` commands below assume the repo
virtualenv is already activated. If it is not activated, use
`.\.venv\Scripts\python.exe`.

## Available Runbooks

- [PoisonedRAG](poisoned_rag.md)
- [Instruction-following](instruction_following.md)
- [Exfiltration](exfiltration.md)

## When To Use Which

- `poisoned_rag.md`: staged PoisonedRAG pipeline based on the selected finalized close-ended dataset
- `instruction_following.md`: indirect instruction-following attack built from the selected finalized clean close-ended set
- `exfiltration.md`: indirect exfiltration attack and matched control corpus built from the selected finalized clean close-ended set

## Canonical Preparation

All attack families should start from the same reviewed close-ended query set.
That shared preparation flow exists to ensure the evaluation questions are
answerable before any malicious documents are introduced.

The intended order is:

1. complete the full clean-data workflow in [../data/README.md](../data/README.md)
2. derive each attack corpus from `data/processed/clean/<dataset>/final/`
3. build attack retrieval artifacts as an append-only derivative of the clean retrieval base
4. run the selected attack experiments

The reduced `close_ended_dev` dataset remains available. Replace the dataset
name in the commands and paths when you want the dev variant instead of the
full-corpus main variant.

For the default `close_ended_main` workflow, the attacked retrieval artifacts
append to the clean `clean_complete` retrieval base. For reduced datasets such
as `close_ended_dev`, the attacked retrieval artifacts append to that dataset's
clean final retrieval artifacts.

## Shared Conventions

- processed corpora and query sets live under `data/processed/`
- retrieval artifacts live under `data/built/`
- experiment logs live under `logs/`
- analysis outputs live under `results/`

Use the repo virtualenv for model-backed attack scripts:

```powershell
.\.venv\Scripts\python.exe
```

Use the batch runner for experiments:

```powershell
python scripts/run/run_experiments.py --dataset-profile <preset> --config all --profile main
```

Runner flags and dataset-profile details are documented in
[../run/README.md](../run/README.md).

Analysis commands, result schemas, and the cross-attack thesis builder are
documented in [../analysis/README.md](../analysis/README.md).

## Retrieval-Artifact Compatibility Settings

Warning: `--tokenizer-name`, `--chunk-size`, `--overlap`, and the effective
input corpus/order are retrieval-artifact compatibility settings. If you
changed any of these when building the clean dataset, you must use the same
values when building attack artifacts. Otherwise, the attacked artifacts may no
longer be an append-only derivative of the clean retrieval base.

Output-path settings such as `--chunks-path`, `--index-path`, and `--meta-path`
change destinations only. They must point to attack-specific artifact
locations, not to the clean retrieval base.

`--device` and `--batch-size` are performance settings rather than structural
compatibility settings. They should not change the logical corpus structure,
but CPU and CUDA runs may not guarantee bit-for-bit identical embeddings.
