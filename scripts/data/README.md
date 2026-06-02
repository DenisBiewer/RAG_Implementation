# Data Preparation

This directory contains the scripts that prepare the clean HotpotQA datasets
used by the rest of the repository.

Unless noted otherwise, the `python ...` commands below assume the repo
virtualenv is already activated. If it is not activated, use
`.\.venv\Scripts\python.exe`.

The intended flow is:

1. build the complete clean HotpotQA corpus and query pool
2. choose `close_ended_main` or `close_ended_dev`
3. build the initial close-ended candidate set
4. run clean baseline validation on that initial set
5. export the script-validated subset and manual-review file
6. review the surviving queries by hand
7. finalize the reviewed close-ended dataset
8. use that finalized dataset for clean runs or as the starting point for an
   attack runbook

## Dataset Choice

Use `close_ended_main` when you want the canonical thesis workflow:

- full clean corpus
- 20 final reviewed target queries
- clean experiment runs reuse the `clean_complete` retrieval artifacts

Use `close_ended_dev` when you want a smaller iteration dataset:

- reduced clean corpus
- 10 final reviewed target queries
- clean experiment runs use dataset-stage-specific retrieval artifacts

## Methodology Notes

| Aspect | Implemented behavior | Source |
| --- | --- | --- |
| HotpotQA split usage | The clean corpus is built from HotpotQA `fullwiki` train and validation contexts. The raw query pool is built from the validation split only. | `scripts/data/build_hotpot_data.py` |
| Clean corpus construction | Context entries are converted to plain text by joining non-empty sentences. Documents are deduplicated by `doc_id = wiki::<title>` and sorted before writing. | `scripts/data/build_hotpot_data.py` |
| Initial close-ended selection | `scripts/data/build_close_ended_query_set.py` applies conservative filtering: duplicate normalized questions are removed; question prefixes are restricted to conservative close-ended forms; answer shapes must remain short and unambiguous; support titles must exist; and non-boolean answers must appear in the supporting documents. | `scripts/data/build_close_ended_query_set.py` |
| Question-form filtering | Supported prefixes include `who`, `what`, `where`, `when`, `which`, `how many`, and `in what year`. Boolean questions are restricted to boolean-style prefixes. `why`, explanatory, and comparison-style forms are rejected. | `scripts/data/build_close_ended_query_set.py` |
| Answer-shape filtering | The selector rejects empty answers, time ranges, list-like answers, excessive punctuation, answers with `or`, overly long spans, and other ambiguous forms. Boolean answers are handled separately. | `scripts/data/build_close_ended_query_set.py` |
| Support-title checks | Queries without support titles are rejected. For registered close-ended datasets, support titles must be present in the local clean corpus. | `scripts/data/build_close_ended_query_set.py`, `app/dataset_registry.py` |
| Answer-in-support check | For non-boolean answers, the normalized gold answer must appear in at least one supporting document before the query can enter the initial close-ended set. | `scripts/data/build_close_ended_query_set.py` |
| Script validation | `validation_pass` is defined as `strict_rule_answer_match and all_support_titles_retrieved`. The check is computed from the clean validation log and evaluates the `final_output`, not the raw `model_output`. | `scripts/analysis/analyze_clean_dataset.py`, `scripts/data/validate_close_ended_query_set.py` |
| Manual review | The reviewer checks whether each script-validated row should remain in the final dataset, can add answer variants through `acceptable_answers`, and must set `manual_validation_pass` explicitly to `true` or `false`. | `scripts/data/validate_close_ended_query_set.py`, `scripts/data/finalize_close_ended_query_set.py` |
| Finalization | If more rows pass manual review than the dataset target count, finalization uses a stable seeded hash selection rather than manual tie-breaking. The effective seed resolves through `app/seed_config.py`; the final selection namespace is `finalize_close_ended_query_set:manual_review_final_targets`. | `scripts/data/finalize_close_ended_query_set.py`, `app/seed_config.py` |
| Attack-family starting point | All attack families begin from `data/processed/clean/<dataset>/final/queries.jsonl` and `corpus.jsonl`, not from the initial candidate set. | `scripts/attacks/*.md` |

## 1. Build the Complete Clean HotpotQA Data

This is the raw ingestion step. It loads HotpotQA `fullwiki`, extracts the
clean corpus, and writes the full evaluation query pool.

```powershell
python scripts/data/build_hotpot_data.py
```

Outputs:

- `data/processed/clean/complete/corpus.jsonl`
- `data/processed/clean/complete/queries.jsonl`

Build the retrieval artifacts for that complete clean corpus as well:

```powershell
python scripts/indexing/build_retrieval_artifacts.py --preset clean_complete
```

Those artifacts are required for `close_ended_main` validation and final clean
runs because `close_ended_main` reuses the complete clean index instead of
building a second full-corpus clean index.

## 2. Build the Initial Close-Ended Dataset

Pick one dataset name:

- `close_ended_main`
- `close_ended_dev`

Then build the initial close-ended candidate set:

```powershell
python scripts/data/build_close_ended_query_set.py --dataset close_ended_main
```

or:

```powershell
python scripts/data/build_close_ended_query_set.py --dataset close_ended_dev
```

Outputs under `data/processed/clean/<dataset>/initial/`:

- `queries.jsonl`
- `corpus.jsonl`
- `accepted_candidates.jsonl`
- `rejected_candidates.jsonl`
- `manifest.json`

If you are using `close_ended_dev`, build the initial-stage retrieval artifacts
before validation:

```powershell
python scripts/indexing/build_retrieval_artifacts.py --preset close_ended_dev_initial
```

You do not need a `close_ended_main_initial` indexing step because that dataset
uses the `clean_complete` retrieval artifacts.

## 3. Run Clean Validation on the Initial Dataset

Run the clean baseline on the initial close-ended dataset. That run is the
input to the validation/export step.

Canonical main-dataset validation run:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile close_ended_main_initial `
  --config baseline `
  --profile main
```

Reduced dev-dataset validation run:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile close_ended_dev_initial `
  --config baseline `
  --profile main
```

Outputs:

- `logs/clean/<dataset>/initial/baseline_runs.jsonl`

## 4. Export Script-Validated Queries and Prepare Manual Review

Convert the clean validation log into:

- a query subset that passed script validation
- a query subset that failed script validation
- a manual-review artifact for the script-validated rows

Main dataset:

```powershell
python scripts/data/validate_close_ended_query_set.py `
  --dataset close_ended_main `
  --config baseline
```

Dev dataset:

```powershell
python scripts/data/validate_close_ended_query_set.py `
  --dataset close_ended_dev `
  --config baseline
```

Outputs:

- `data/processed/clean/<dataset>/script_validated/validated_queries.jsonl`
- `data/processed/clean/<dataset>/script_validated/failed_validation_queries.jsonl`
- `data/processed/clean/<dataset>/script_validated/validation_manifest.json`
- `data/processed/clean/<dataset>/final/manual_review.jsonl`

## 5. Complete Manual Review

Open:

- `data/processed/clean/<dataset>/final/manual_review.jsonl`

For every row, set:

- `manual_validation_pass` to `true` or `false`
- `acceptable_answers` if the expected answer needs acceptable variants
- optionally `reason`

Interpretation:

- `manual_validation_pass = true`: retain the row for possible final selection
- `manual_validation_pass = false`: exclude the row from the final dataset
- `acceptable_answers`: approved answer variants carried into the finalized
  query file for downstream evaluation

Finalization requires an explicit `true` or `false` decision for every review
row. Leaving rows undecided will stop the finalization step.

## 6. Finalize the Reviewed Close-Ended Dataset

Once manual review is complete, finalize the clean close-ended dataset:

```powershell
python scripts/data/finalize_close_ended_query_set.py --dataset close_ended_main
```

or:

```powershell
python scripts/data/finalize_close_ended_query_set.py --dataset close_ended_dev
```

Outputs under `data/processed/clean/<dataset>/final/`:

- `queries.jsonl`
- `corpus.jsonl`
- `manifest.json`

If you are using `close_ended_dev`, build the final-stage retrieval artifacts
after finalization:

```powershell
python scripts/indexing/build_retrieval_artifacts.py --preset close_ended_dev_final
```

You do not need a `close_ended_main_final` indexing step because
`close_ended_main` again reuses the `clean_complete` retrieval artifacts.

## 7. Run Clean Final Experiments

Canonical clean thesis runs:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile close_ended_main_final `
  --config all `
  --profile main
```

Reduced clean iteration runs:

```powershell
python scripts/run/run_experiments.py `
  --dataset-profile close_ended_dev_final `
  --config all `
  --profile main
```

## 8. Move to an Attack Workflow

All attack families should start from the finalized clean close-ended dataset:

- `data/processed/clean/<dataset>/final/queries.jsonl`
- `data/processed/clean/<dataset>/final/corpus.jsonl`

Then follow the dedicated runbook:

- [PoisonedRAG](../attacks/poisoned_rag.md)
- [Instruction-following](../attacks/instruction_following.md)
- [Exfiltration](../attacks/exfiltration.md)
