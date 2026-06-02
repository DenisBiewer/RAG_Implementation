# Scripts

This directory is organized by purpose:

- `data/`: dataset creation, validation, manual-review preparation, and finalization
- `indexing/`: chunking and FAISS index construction
- `attacks/`: attack-corpus builders and attack-specific workflow docs
- `analysis/`: experiment-log analysis and evaluation
- `inspect/`: inspection and debugging helpers
- `run/`: experiment runners

Use the scripts from their grouped locations directly, for example:

- `python scripts/data/build_close_ended_query_set.py`
- `python scripts/data/validate_close_ended_query_set.py`
- `python scripts/indexing/build_retrieval_artifacts.py --preset instruction_following_close_ended_main`
- `python scripts/run/run_experiments.py --dataset-profile close_ended_main_final --config baseline --profile main`

Unless noted otherwise, `python ...` examples in this directory assume the repo
virtualenv is already activated. If it is not activated, use
`.\.venv\Scripts\python.exe`.

## Canonical Workflow

If you want the intended project flow from start to finish, use this order:

1. build the complete clean HotpotQA dataset with `scripts/data/build_hotpot_data.py`
2. build `clean_complete` retrieval artifacts
3. choose `close_ended_main` or `close_ended_dev`
4. build the initial close-ended candidate set
5. run clean baseline validation on that initial set
6. export the script-validated subset and prepare `manual_review.jsonl`
7. complete manual review
8. finalize the reviewed close-ended dataset
9. run clean final experiments or continue into an attack-specific runbook

The exact command sequence and methodology notes for those steps live in
[data/README.md](data/README.md).

## Dataset Variants

The repository's main close-ended workflows are:

1. `close_ended_main`: full-corpus final dataset for canonical thesis runs
2. `close_ended_dev`: reduced clean dataset for faster iteration

Both variants use the same review pipeline:

1. build an initial close-ended candidate set
2. validate it with clean baseline runs
3. export script-validated/manual-review artifacts
4. finalize the reviewed dataset
5. use that finalized dataset for clean runs and all attack families

For `close_ended_main`, clean validation and final clean runs reuse the
`clean_complete` retrieval artifacts rather than building a second full-corpus
clean index. `close_ended_dev` uses its own stage-specific retrieval artifacts.

## Retrieval-Artifact Compatibility Settings

When you build attack retrieval artifacts, the attacked artifacts are expected
to be an append-only derivative of an existing clean retrieval base.

Warning: `--tokenizer-name`, `--chunk-size`, `--overlap`, and the effective
input corpus/order are retrieval-artifact compatibility settings. If you
changed any of these when building the clean dataset, you must use the same
values when building attack artifacts. Otherwise, the attacked artifacts may no
longer be an append-only derivative of the clean retrieval base.

Output-path settings such as `--chunks-path`, `--index-path`, and `--meta-path`
change where artifacts are written. They should point to attack-specific output
locations and must not overwrite clean artifacts.

`--device` and `--batch-size` are performance settings rather than structural
compatibility settings. They should not change the logical corpus structure,
but CPU and CUDA runs may not guarantee bit-for-bit identical embeddings.

## Workflow Docs

Attack-specific runbooks live under [attacks/README.md](attacks/README.md):

- [PoisonedRAG](attacks/poisoned_rag.md)
- [Instruction-following](attacks/instruction_following.md)
- [Exfiltration](attacks/exfiltration.md)

Experiment-runner usage, dataset presets, canonical config names, and
reproducibility notes live in [run/README.md](run/README.md).

Analysis usage, generated outputs, metric definitions, and thesis-table
generation live in [analysis/README.md](analysis/README.md).

## Typical Navigation

If you are preparing data:

- start with [data/README.md](data/README.md)
- build raw HotpotQA outputs with `scripts/data/build_hotpot_data.py`
- use `scripts/data/validate_close_ended_query_set.py` to export the
  script-validated subset and prepare the manual-review artifact
- complete manual review before calling `scripts/data/finalize_close_ended_query_set.py`
- build retrieval artifacts from `scripts/indexing/` for reduced clean datasets
  and attack datasets
- reuse `clean_complete` retrieval artifacts for clean `close_ended_main` runs

If you are running attacks:

- first prepare the finalized close-ended clean set through the full workflow in
  [data/README.md](data/README.md)
- follow the dedicated runbook in `scripts/attacks/`
- use `scripts/run/run_experiments.py` to execute the selected configs
- use `scripts/analysis/` to summarize the resulting logs and generate
  thesis-facing tables
