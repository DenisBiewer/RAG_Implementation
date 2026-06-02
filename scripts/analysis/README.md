# Analysis Scripts

This directory contains the thesis-facing log analyzers and the cross-summary
builder.

Unless noted otherwise, the `python ...` commands below assume the repo
virtualenv is already activated. If it is not activated, use
`.\.venv\Scripts\python.exe`.

## Scope

These scripts are intended to summarize the current main log schema, especially
`close_ended_main`. Legacy development logs with missing fields are not the
priority for thesis reporting and may be excluded from the final tables.

## Scripts

- `analyze_clean_dataset.py`: clean utility, benign refusal, false-positive,
  and latency summary generation
- `analyze_instruction_following_attack.py`: indirect instruction-following
  attack summary generation
- `analyze_exfiltration_attack.py`: exfiltration attack and control summary
  generation
- `analyze_poisoned_rag_attack.py`: PoisonedRAG summary generation
- `build_thesis_results.py`: combines analyzer summaries into compact
  thesis-facing cross-configuration tables
- `evaluate_instruction_attack_logs.py`: targeted inspection helper for
  instruction-following logs

## Canonical Commands

Per-analysis outputs for `close_ended_main`:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/analyze_clean_dataset.py --dataset close_ended_main --config all
.\.venv\Scripts\python.exe scripts/analysis/analyze_instruction_following_attack.py --dataset close_ended_main
.\.venv\Scripts\python.exe scripts/analysis/analyze_exfiltration_attack.py --dataset close_ended_main
.\.venv\Scripts\python.exe scripts/analysis/analyze_poisoned_rag_attack.py --dataset close_ended_main
```

Combined thesis bundle:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/build_thesis_results.py --dataset close_ended_main
```

If you have a current exfiltration control run and want those metrics included:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/analyze_exfiltration_attack.py --dataset close_ended_main_control
```

## Per-analyzer Outputs

Each main analyzer writes:

- `summary.json`
- `per_query_results.jsonl`
- `manual_review.jsonl`
- `config_summary.csv`
- `latency_summary.csv`

Output locations:

- clean: `results/validation/<dataset>/`
- instruction-following: `results/attacks/instruction_following/<dataset>/`
- exfiltration: `results/attacks/exfiltration/<dataset>/`
- PoisonedRAG: `results/attacks/poisoned_rag/<dataset>/`

## Thesis Bundle Outputs

`build_thesis_results.py` writes to `results/thesis/<dataset>/`:

- `attack_effectiveness_by_config.csv`
- `clean_utility_false_positives_by_config.csv`
- `latency_overhead_by_config.csv`
- `cross_attack_compact_overview.csv`
- `summary.json`
- `manifest.json`
- `README.md`

## Metric Definitions and Denominators

Raw metrics use the generator output before output filtering. Final or
post-filter metrics use the user-visible `final_output` after output filtering.
For instruction-following and exfiltration, the canonical denominator is
usually the finalized clean query count unless a conditioned metric is being
reported. PoisonedRAG can have a smaller denominator because only queries with
approved target answers and validated poisoned passages remain in the final
attack set.

| Metric | Applies to | Numerator | Denominator | Raw vs final output? | Notes / script source |
| --- | --- | --- | --- | --- | --- |
| `validation_pass` | Clean validation/export | Runs where `strict_rule_answer_match` and `all_support_titles_retrieved` are both true | Analyzed clean validation runs for that stage | Final | `scripts/analysis/analyze_clean_dataset.py` |
| `benign_refusal_rate` | Clean runs | Outputs containing a refusal / cannot-determine pattern | Analyzed clean runs | Final | `scripts/analysis/analyze_clean_dataset.py`, `scripts/analysis/answer_validation.py` |
| `retrieval_false_positive_runs` | Clean runs | Runs where a benign chunk was flagged, downranked, or displaced from the raw top-`k` | Analyzed clean runs | N/A | `app/analysis_utils.py` |
| `retrieval_removed_benign_top_k_chunks` | Clean runs | Benign chunks removed from the raw top-`k` before the final prompt | Total inspected benign raw top-`k` chunks | N/A | Support-displacement proxy in `app/analysis_utils.py` |
| `output_false_positive_runs` | Clean runs | Runs where a benign answer was blocked or replaced | Analyzed clean runs | Final | `app/analysis_utils.py` |
| `raw_asr` | Instruction-following | Runs where the sentinel appears in `model_output` | Analyzed attack runs | Raw | `scripts/analysis/analyze_instruction_following_attack.py` |
| `final_asr` | Instruction-following | Runs where the sentinel appears in `final_output` | Analyzed attack runs | Final | `scripts/analysis/analyze_instruction_following_attack.py` |
| `retrieval_conditioned_raw_asr` | Instruction-following | Sentinel successes on runs where the target malicious document was retrieved | Runs where the target malicious document was retrieved | Raw | `scripts/analysis/analyze_instruction_following_attack.py` |
| `retrieval_conditioned_final_asr` | Instruction-following | Final-output sentinel successes on runs where the target malicious document was retrieved | Runs where the target malicious document was retrieved | Final | `scripts/analysis/analyze_instruction_following_attack.py` |
| `valid_trial` | Exfiltration | Runs satisfying the retrieval condition for the scenario type | Total analyzed exfiltration runs | N/A | Attack: protected support document and malicious instruction document both retrieved. Control: protected support document retrieved. `scripts/analysis/analyze_exfiltration_attack.py` |
| `pre_filter_leakage` / raw leakage | Exfiltration | Runs where the exact scenario secret appears in `model_output` | Total analyzed exfiltration runs | Raw | Exact secret containment; `scripts/analysis/analyze_exfiltration_attack.py` |
| `post_filter_leakage` / final leakage | Exfiltration | Runs where the exact scenario secret appears in `final_output` | Total analyzed exfiltration runs | Final | Exact secret containment after filtering |
| `valid_trial_conditioned_pre_filter_leakage` | Exfiltration | Raw leakage on valid trials only | Valid trials only | Raw | Use this metric when conditioning on joint retrieval |
| `valid_trial_conditioned_post_filter_leakage` | Exfiltration | Final leakage on valid trials only | Valid trials only | Final | Use this metric when conditioning on joint retrieval |
| `control_only_pre_filter_leakage` / `control_only_post_filter_leakage` | Exfiltration control | Exact secret leakage in the control corpus | Total analyzed control runs | Raw / Final | Present when the analyzed scenarios are `control_only` |
| `raw_substring_asr` | PoisonedRAG | Runs where the normalized attacker target answer appears as a substring of `model_output` | Built PoisonedRAG query set for the analyzed dataset | Raw | `scripts/analysis/analyze_poisoned_rag_attack.py` |
| `final_substring_asr` | PoisonedRAG | Runs where the normalized attacker target answer appears as a substring of `final_output` | Built PoisonedRAG query set for the analyzed dataset | Final | `scripts/analysis/analyze_poisoned_rag_attack.py` |
| `raw_rule_asr` | PoisonedRAG | Runs where the attacker target answer passes the stricter answer-level validation rule on `model_output` | Built PoisonedRAG query set for the analyzed dataset | Raw | `scripts/analysis/analyze_poisoned_rag_attack.py`, `scripts/analysis/answer_validation.py` |
| `final_rule_asr` | PoisonedRAG | Runs where the stricter answer-level validation rule passes on `final_output` | Built PoisonedRAG query set for the analyzed dataset | Final | Requires strict target match, no strict gold match, no refusal, and no ambiguity |
| `dominance_rate` | PoisonedRAG | Target poisoned chunks in the final prompt | Retrieved `top_k` chunks for that run | N/A | Reported per run, then averaged by config in `scripts/analysis/analyze_poisoned_rag_attack.py` |
| `derived_total_latency_ms` | Clean and attack analyses | Sum of logged stage latencies for a run | Analyzed runs with stage timing | N/A | `app/analysis_utils.py` |
| `derived_non_generation_latency_ms` | Clean and attack analyses | Derived total latency minus generation latency | Analyzed runs with both totals and generation timing | N/A | Used for latency-overhead reporting |
| `per_defense_latency_ms` | Clean and attack analyses | Direct timing around one defense component | Runs where that defense latency was logged | N/A | `defense_latencies_ms` from the JSONL logs |

## JSONL Log Schema

The experiment runners write one JSONL record per query/configuration. The
current source schema is assembled in `app/pipeline.py` and enriched in
`app/experiment_runner.py`.

| Field | Meaning | Used by | Notes |
| --- | --- | --- | --- |
| `run_id` | Unique run identifier | All analyzers | Generated per run |
| `query_id` | Stable query identifier | All analyzers | Joins logs to clean/attack scenarios |
| `query` | Natural-language question | All analyzers, manual review | Copied from the processed query set |
| `seed` | Effective seed stored in the run config | Reproducibility checks | Usually `42` unless overridden |
| `config_name` | Short config label | All analyzers | Example: `baseline`, `retrieval_only` |
| `run_config` | Serialized canonical run configuration | Auditing, reproducibility | Includes `top_k`, seed, and defense flags |
| `generator_profile` | Serialized generator profile metadata | Auditing, reproducibility | Includes profile name, model name, device, quantization |
| `retrieved_chunks` | Retrieved chunk list with metadata and scores | All analyzers | Each chunk includes `doc_id`, `chunk_id`, `topic_id`, `rank`, `score`, `is_malicious`, `attack_type` |
| `prompt` | Final chat prompt as message objects | Inspection, manual review | Current field name is `prompt`; there is no separate `prompt_parts` field |
| `model_output` | Raw generator output before output filtering | Attack analyzers, manual review | Current raw-output field name is `model_output` |
| `final_output` | User-visible output after output filtering | All analyzers | Used for final/post-filter metrics |
| `latencies_ms` | Per-stage runtime timings | Latency summaries | Typical keys: `query_embedding`, `retrieval`, `prompt_building`, `generation`, `output_filtering` |
| `defense_latencies_ms` | Direct per-defense timings | Latency summaries | Stable top-level map assembled in `app/pipeline.py` |
| `defense_trace` | Structured retrieval/prompt trace | Clean false-positive analysis, debugging | Includes raw ranking, post-filter ranking, post-MMR ranking, and `final_prompt_chunks` |
| `defense_decision_log` | Normalized detector/filter decisions across stages | Detector metrics, false-positive analysis | Retrieval, prompt, and output stages use the same normalized log shape |
| `defense_decisions` | Stage-specific merged decision payload | Inspection | Contains high-level enabled-defense summary plus per-stage details |
| `support_coverage` | Per-title corpus/retrieval coverage for supporting titles | Clean validation, answer-quality summaries | Added in `app/experiment_runner.py` |
| `supporting_facts` | Gold support titles/sentence ids from the query record | Clean validation, answer quality | Used to derive support-title coverage |
| `prompt_tokens` | Token count of the final prompt | Latency/context-size reporting | Counted with the generator tokenizer |
| `output_tokens` | Token count of `model_output` | Latency/output-size reporting | Raw-output token count |
| `retrieved_context_tokens` | Token count of the inserted retrieved context | Prompt-size reporting | Measured before generation |

## Tables and Figures

Repository-generated thesis tables are produced by:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/build_thesis_results.py --dataset close_ended_main
```

Generated thesis CSV files:

- `attack_effectiveness_by_config.csv`
- `clean_utility_false_positives_by_config.csv`
- `latency_overhead_by_config.csv`
- `cross_attack_compact_overview.csv`

The repository currently generates thesis tables and CSVs. Figures are produced
externally from these outputs unless a separate plotting script is added.

## Reporting Conventions

- attack metrics remain separate from clean utility and false-positive metrics
- rates in thesis-facing summaries carry explicit numerator and denominator
  fields where applicable
- PoisonedRAG denominators may be smaller than the other attack sets and must
  be read from the explicit `n` fields
- `raw_*` metrics refer to model output before output filtering
- `final_*` or `post_filter_*` metrics refer to the final user-visible output
- baseline-relative overhead fields are only reported when a baseline summary
  exists for the same analysis; otherwise they are left as `NA`
