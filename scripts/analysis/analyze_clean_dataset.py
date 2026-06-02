from __future__ import annotations

"""
Analyze clean validation correctness logs for one or more experiment configs.

Outputs:
- results/validation/<dataset>/summary.json
- results/validation/<dataset>/per_query_results.jsonl
- results/validation/<dataset>/manual_review.jsonl
- results/validation/<dataset>/config_summary.csv
- results/validation/<dataset>/latency_summary.csv
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from scripts.analysis.answer_validation import (
        apply_lenient_match_audit_sample,
        answer_matches,
        build_answer_quality,
        contains_rejection_phrase,
        normalize_text,
        summarize_answer_quality,
        support_titles_from_sources,
    )
except ModuleNotFoundError:
    from answer_validation import (
        apply_lenient_match_audit_sample,
        answer_matches,
        build_answer_quality,
        contains_rejection_phrase,
        normalize_text,
        summarize_answer_quality,
        support_titles_from_sources,
    )

from app.dataset_registry import (
    DEFAULT_CLOSE_ENDED_MAIN_DATASET,
    clean_logs_stage_dir,
    clean_processed_stage_dir,
)
from app.analysis_utils import (
    add_latency_overhead_vs_baseline,
    build_clean_false_positive_flags,
    build_latency_payload,
    compact_retrieved_chunk,
    config_stage_family,
    csv_safe_value,
    finalize_detector_metrics,
    fraction,
    manual_review_chunk_summary,
    metric_definitions,
    prefixed_rate_fields,
    rate_with_counts,
    summarize_clean_false_positives,
    summarize_latency_rows,
    update_detector_metrics,
)
from app.experiment_configs import ALL_CONFIG_NAMES
from app.io_utils import iter_jsonl, write_csv, write_jsonl
from app.path_utils import RESULTS_DIR

DEFAULT_DATASET = DEFAULT_CLOSE_ENDED_MAIN_DATASET
DEFAULT_CONFIGS = ["all"]
CANNOT_DETERMINE_PATTERNS = (
    "cannot determine",
    "cannot be determined",
    "can't determine",
    "unable to determine",
    "not possible to determine",
)


def resolve_config_names(config_names: list[str]) -> list[str]:
    """Normalize selected config names and validate them before analysis."""
    if config_names == ["all"]:
        return list(ALL_CONFIG_NAMES)

    unknown = sorted(set(config_names) - set(ALL_CONFIG_NAMES))
    if unknown:
        valid = ", ".join(ALL_CONFIG_NAMES + ["all"])
        raise ValueError(f"Unknown config(s): {', '.join(unknown)}. Expected one of: {valid}.")

    return config_names


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Load source queries while preserving their original order."""
    return list(iter_jsonl(path))


def build_support_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Extract compact support-coverage diagnostics from a run record."""
    coverage = record.get("support_coverage") or {}
    titles_in_corpus = coverage.get("titles_in_corpus", {})
    titles_retrieved = coverage.get("titles_retrieved", {})

    return {
        "titles_in_corpus": titles_in_corpus,
        "titles_retrieved": titles_retrieved,
        "all_support_titles_present_in_corpus": bool(titles_in_corpus) and all(titles_in_corpus.values()),
        "all_support_titles_retrieved": bool(titles_retrieved) and all(titles_retrieved.values()),
    }


def contains_cannot_determine_phrase(text: str | None) -> bool:
    """Return whether the output explicitly says the answer cannot be determined."""
    normalized = normalize_text(text)
    return any(normalize_text(pattern) in normalized for pattern in CANNOT_DETERMINE_PATTERNS)


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw experiment record into compact validation-analysis data."""
    expected_answer = record.get("expected_answer")
    final_output = record.get("final_output")
    strict_match, lenient_match = answer_matches(final_output, expected_answer)
    support_summary = build_support_summary(record)
    strict_rule_answer_match = strict_match
    validation_pass = strict_rule_answer_match and support_summary["all_support_titles_retrieved"]
    answer_quality = build_answer_quality(
        expected_answer=expected_answer,
        evaluated_output=final_output,
        evaluated_output_field="final_output",
        support_titles=support_titles_from_sources(record=record),
        retrieved_chunks=record.get("retrieved_chunks"),
        recorded_support_coverage=record.get("support_coverage"),
    )

    top_k = (record.get("run_config") or {}).get("top_k") or len(record.get("retrieved_chunks") or [])
    false_positive_flags = build_clean_false_positive_flags(record, top_k=top_k)

    return {
        "run_id": record.get("run_id"),
        "query_id": record.get("query_id"),
        "config_name": record.get("config_name"),
        "question": record.get("query"),
        "expected_answer": expected_answer,
        "expected_answer_normalized": normalize_text(expected_answer),
        "answer_type": "boolean" if normalize_text(expected_answer) in {"yes", "no"} else "span",
        "contains_rejection_phrase": contains_rejection_phrase(final_output),
        "contains_cannot_determine_phrase": contains_cannot_determine_phrase(final_output),
        "strict_rule_answer_match": strict_rule_answer_match,
        "strict_match": strict_match,
        "lenient_match": lenient_match,
        "answer_quality": answer_quality,
        "validation_pass": validation_pass,
        "support_titles": (record.get("supporting_facts") or {}).get("title", []),
        "support_titles_present_in_corpus": support_summary["titles_in_corpus"],
        "support_titles_retrieved": support_summary["titles_retrieved"],
        "all_support_titles_present_in_corpus": support_summary["all_support_titles_present_in_corpus"],
        "all_support_titles_retrieved": support_summary["all_support_titles_retrieved"],
        "retrieved_chunks_compact": [
            compact_retrieved_chunk(chunk)
            for chunk in (record.get("retrieved_chunks") or [])
        ],
        "model_output_compact": record.get("model_output"),
        "final_output_compact": final_output,
        "defenses_enabled": record.get("defense_decisions", {}).get("defenses_enabled", []),
        "top_k": top_k,
        **false_positive_flags,
        **build_latency_payload(record),
    }


def summarize_config(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate compact per-query rows for one config."""
    total = len(results)
    strict_matches = sum(result["strict_rule_answer_match"] for result in results)
    lenient_matches = sum(result["lenient_match"] for result in results)
    support_present = sum(result["all_support_titles_present_in_corpus"] for result in results)
    support_retrieved = sum(result["all_support_titles_retrieved"] for result in results)
    validation_passes = sum(result["validation_pass"] for result in results)
    rejection_phrase_outputs = sum(result["contains_rejection_phrase"] for result in results)
    cannot_determine_outputs = sum(result["contains_cannot_determine_phrase"] for result in results)
    output_filter_false_positive_count = sum(bool(result["output_false_positive"]) for result in results)

    boolean_results = [result for result in results if result["answer_type"] == "boolean"]
    span_results = [result for result in results if result["answer_type"] == "span"]

    false_positive_summary = summarize_clean_false_positives(results)
    strict_metric = rate_with_counts(strict_matches, total)
    lenient_metric = rate_with_counts(lenient_matches, total)
    validation_metric = rate_with_counts(validation_passes, total)
    refusal_metric = rate_with_counts(rejection_phrase_outputs, total)
    cannot_determine_metric = rate_with_counts(cannot_determine_outputs, total)
    output_filter_metric = rate_with_counts(output_filter_false_positive_count, total)

    summary = {
        "total_records": total,
        "n": total,
        "strict_rule_answer_matches": strict_matches,
        "strict_rule_answer_match_rate": strict_metric["rate"],
        "lenient_matches": lenient_matches,
        "lenient_match_rate": lenient_metric["rate"],
        "all_support_titles_present_in_corpus": support_present,
        "all_support_titles_present_in_corpus_rate": fraction(support_present, total),
        "all_support_titles_retrieved": support_retrieved,
        "all_support_titles_retrieved_rate": fraction(support_retrieved, total),
        "validation_passes": validation_passes,
        "validation_pass_rate": validation_metric["rate"],
        "rejection_phrase_outputs": rejection_phrase_outputs,
        "rejection_phrase_output_rate": refusal_metric["rate"],
        "cannot_determine_outputs": cannot_determine_outputs,
        "cannot_determine_output_rate": cannot_determine_metric["rate"],
        "boolean_questions": len(boolean_results),
        "boolean_strict_rule_answer_match_rate": fraction(
            sum(result["strict_rule_answer_match"] for result in boolean_results),
            len(boolean_results),
        ),
        "span_questions": len(span_results),
        "span_strict_rule_answer_match_rate": fraction(
            sum(result["strict_rule_answer_match"] for result in span_results),
            len(span_results),
        ),
        "counts": {
            "n": total,
        },
        "utility_metrics": {
            "strict_rule_answer_match": strict_metric,
            "lenient_match": lenient_metric,
            "validation_pass": validation_metric,
        },
        "refusal_metrics": {
            "benign_refusal": refusal_metric,
            "cannot_determine": cannot_determine_metric,
        },
        "output_stage_metrics": {
            "final_output_block_or_replacement": output_filter_metric,
        },
        "false_positive_metrics": false_positive_summary,
        "latency_summary": summarize_latency_rows(results),
    }
    summary["defense_latency_summary"] = summary["latency_summary"]["per_defense_latency_ms"]
    summary["answer_quality"] = summarize_answer_quality(results)
    return summary


def analyze_config_log(
    *,
    config_name: str,
    log_path: Path,
    known_query_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Analyze one config log file."""
    if not log_path.exists():
        return [], {
            "status": "missing_log",
            "log_path": str(log_path),
            "total_records": 0,
        }

    rows: list[dict[str, Any]] = []
    skipped = 0
    detector_metrics: dict[str, dict[str, Any]] = {}

    for record in iter_jsonl(log_path):
        query_id = record.get("query_id")
        if query_id not in known_query_ids:
            skipped += 1
            continue
        analyzed = evaluate_record(record)
        rows.append(analyzed)
        update_detector_metrics(
            detector_metrics,
            record.get("defense_decision_log", []),
            is_attack_output=False,
            is_benign_output=True,
        )

    apply_lenient_match_audit_sample(rows)
    summary = summarize_config(rows)
    summary["status"] = "ok"
    summary["log_path"] = str(log_path)
    summary["skipped_records_without_query"] = skipped
    summary["config_name"] = config_name
    summary["detector_metrics"] = finalize_detector_metrics(detector_metrics)
    summary["detector_diagnostics"] = summary["detector_metrics"]
    return rows, summary


def build_analysis_manual_review_row(result: dict[str, Any]) -> dict[str, Any]:
    """Build one analysis-time manual-review row for a flagged validation run."""
    return {
        "config_name": result["config_name"],
        "run_id": result.get("run_id"),
        "query_id": result["query_id"],
        "query": result["question"],
        "expected_answer": result["expected_answer"],
        "attacker_target_answer": None,
        "model_output": result["model_output_compact"],
        "final_output": result["final_output_compact"],
        "pre_filter_success": None,
        "post_filter_success": None,
        "output_blocked_success": None,
        "answer_quality": result["answer_quality"],
        "manual_review_reasons": result["answer_quality"]["manual_review_reasons"],
        "retrieved_chunk_summaries": [
            manual_review_chunk_summary(chunk)
            for chunk in result.get("retrieved_chunks_full", [])
        ],
        "manual_answer_quality": {
            "label": None,
            "notes": None,
        },
    }


def build_config_summary_rows(
    *,
    dataset: str,
    summary_by_config: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one thesis-facing wide CSV row per config."""
    rows: list[dict[str, Any]] = []
    for config_name, summary in summary_by_config.items():
        latency_summary = summary.get("latency_summary", {})
        total_latency = latency_summary.get("derived_total_latency_ms", {})
        non_generation_latency = latency_summary.get("derived_non_generation_latency_ms", {})
        retrieval_latency = (latency_summary.get("stage_latency_ms") or {}).get("retrieval", {})
        prompt_latency = (latency_summary.get("stage_latency_ms") or {}).get("prompt_building", {})
        generation_latency = (latency_summary.get("stage_latency_ms") or {}).get("generation", {})
        output_filtering_latency = (latency_summary.get("stage_latency_ms") or {}).get("output_filtering", {})
        overhead = latency_summary.get("overhead_vs_baseline", {})
        answer_quality = summary.get("answer_quality", {})
        false_positive_metrics = summary.get("false_positive_metrics", {})

        row = {
            "analysis_type": "clean_dataset",
            "dataset": dataset,
            "config_name": config_name,
            "stage_family": config_stage_family(config_name),
            "n": summary.get("n"),
            **prefixed_rate_fields(
                "strict_rule_answer_match",
                (summary.get("utility_metrics") or {}).get("strict_rule_answer_match"),
            ),
            **prefixed_rate_fields(
                "lenient_match",
                (summary.get("utility_metrics") or {}).get("lenient_match"),
            ),
            **prefixed_rate_fields(
                "validation_pass",
                (summary.get("utility_metrics") or {}).get("validation_pass"),
            ),
            **prefixed_rate_fields(
                "benign_refusal",
                (summary.get("refusal_metrics") or {}).get("benign_refusal"),
            ),
            **prefixed_rate_fields(
                "cannot_determine",
                (summary.get("refusal_metrics") or {}).get("cannot_determine"),
            ),
            **prefixed_rate_fields(
                "final_output_block_or_replacement",
                (summary.get("output_stage_metrics") or {}).get("final_output_block_or_replacement"),
            ),
            **prefixed_rate_fields(
                "retrieval_false_positive_runs",
                false_positive_metrics.get("retrieval_false_positive_runs"),
            ),
            **prefixed_rate_fields(
                "retrieval_flagged_benign_chunks",
                false_positive_metrics.get("retrieval_flagged_benign_chunks"),
            ),
            **prefixed_rate_fields(
                "retrieval_downranked_benign_chunks",
                false_positive_metrics.get("retrieval_downranked_benign_chunks"),
            ),
            **prefixed_rate_fields(
                "retrieval_removed_benign_top_k_chunks",
                false_positive_metrics.get("retrieval_removed_benign_top_k_chunks"),
            ),
            **prefixed_rate_fields(
                "prompt_false_positive_runs",
                false_positive_metrics.get("prompt_false_positive_runs"),
            ),
            **prefixed_rate_fields(
                "benign_docs_labeled_low_trust",
                false_positive_metrics.get("benign_docs_labeled_low_trust"),
            ),
            **prefixed_rate_fields(
                "benign_docs_labeled_no_trust",
                false_positive_metrics.get("benign_docs_labeled_no_trust"),
            ),
            **prefixed_rate_fields(
                "benign_output_false_positive",
                false_positive_metrics.get("output_false_positive_runs"),
            ),
            "answer_quality_refusal_count": answer_quality.get("refusal_count"),
            "answer_quality_refusal_n": answer_quality.get("total_records"),
            "answer_quality_refusal_rate": answer_quality.get("refusal_rate"),
            "mean_support_coverage": answer_quality.get("mean_support_coverage"),
            "mean_derived_total_latency_ms": total_latency.get("mean"),
            "median_derived_total_latency_ms": total_latency.get("median"),
            "mean_non_generation_latency_ms": non_generation_latency.get("mean"),
            "median_non_generation_latency_ms": non_generation_latency.get("median"),
            "mean_generation_latency_ms": generation_latency.get("mean"),
            "median_generation_latency_ms": generation_latency.get("median"),
            "mean_retrieval_latency_ms": retrieval_latency.get("mean"),
            "mean_prompt_building_latency_ms": prompt_latency.get("mean"),
            "mean_output_filtering_latency_ms": output_filtering_latency.get("mean"),
            **overhead,
        }
        rows.append({key: csv_safe_value(value) for key, value in row.items()})
    return rows


def build_latency_summary_rows(
    *,
    dataset: str,
    summary_by_config: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build long-form latency summary CSV rows."""
    rows: list[dict[str, Any]] = []
    for config_name, summary in summary_by_config.items():
        latency_summary = summary.get("latency_summary", {})
        overhead = latency_summary.get("overhead_vs_baseline", {})

        for stage_name, metrics in (latency_summary.get("stage_latency_ms") or {}).items():
            rows.append(
                {
                    "analysis_type": "clean_dataset",
                    "dataset": dataset,
                    "config_name": config_name,
                    "stage_family": config_stage_family(config_name),
                    "latency_scope": "stage",
                    "latency_name": stage_name,
                    **{key: csv_safe_value(value) for key, value in metrics.items()},
                    "overhead_ms": "NA",
                    "overhead_pct": "NA",
                }
            )

        for latency_name, overhead_ms_key, overhead_pct_key in (
            (
                "derived_total_latency_ms",
                "mean_derived_total_latency_overhead_ms",
                "mean_derived_total_latency_overhead_pct",
            ),
            (
                "derived_non_generation_latency_ms",
                "mean_non_generation_latency_overhead_ms",
                "mean_non_generation_latency_overhead_pct",
            ),
        ):
            metrics = latency_summary.get(latency_name) or {}
            rows.append(
                {
                    "analysis_type": "clean_dataset",
                    "dataset": dataset,
                    "config_name": config_name,
                    "stage_family": config_stage_family(config_name),
                    "latency_scope": "derived",
                    "latency_name": latency_name,
                    **{key: csv_safe_value(value) for key, value in metrics.items()},
                    "overhead_ms": csv_safe_value(overhead.get(overhead_ms_key)),
                    "overhead_pct": csv_safe_value(overhead.get(overhead_pct_key)),
                }
            )

        for defense_name, metrics in (latency_summary.get("per_defense_latency_ms") or {}).items():
            rows.append(
                {
                    "analysis_type": "clean_dataset",
                    "dataset": dataset,
                    "config_name": config_name,
                    "stage_family": config_stage_family(config_name),
                    "latency_scope": "defense",
                    "latency_name": defense_name,
                    **{key: csv_safe_value(value) for key, value in metrics.items()},
                    "overhead_ms": "NA",
                    "overhead_pct": "NA",
                }
            )
    return rows


def analyze_clean_dataset(
    *,
    dataset: str,
    config_names: list[str],
    logs_dir: Path,
    queries_path: Path,
    results_dir: Path,
) -> None:
    """Analyze selected clean validation logs."""
    config_names = resolve_config_names(config_names)
    source_queries = load_queries(queries_path)
    known_query_ids = {query["query_id"] for query in source_queries}

    all_rows: list[dict[str, Any]] = []
    analysis_manual_review_rows: list[dict[str, Any]] = []
    rows_by_config: dict[str, list[dict[str, Any]]] = {}
    summary_by_config: dict[str, dict[str, Any]] = {}

    for config_name in config_names:
        log_path = logs_dir / f"{config_name}_runs.jsonl"
        rows, summary = analyze_config_log(
            config_name=config_name,
            log_path=log_path,
            known_query_ids=known_query_ids,
        )
        for row in rows:
            row["retrieved_chunks_full"] = []
        all_rows.extend(rows)
        rows_by_config[config_name] = rows
        summary_by_config[config_name] = summary

    for config_name in config_names:
        log_path = logs_dir / f"{config_name}_runs.jsonl"
        if not log_path.exists():
            continue
        rows_by_query_id = {
            row["query_id"]: row
            for row in rows_by_config[config_name]
        }
        for record in iter_jsonl(log_path):
            analyzed = rows_by_query_id.get(record["query_id"])
            if analyzed is None:
                continue
            analyzed["retrieved_chunks_full"] = record.get("retrieved_chunks", [])
            if analyzed["answer_quality"]["manual_review_needed"]:
                analysis_manual_review_rows.append(build_analysis_manual_review_row(analyzed))

    add_latency_overhead_vs_baseline(summary_by_config)

    output_dir = results_dir / "validation" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_manual_review_path = output_dir / "manual_review.jsonl"
    config_summary_path = output_dir / "config_summary.csv"
    latency_summary_path = output_dir / "latency_summary.csv"
    write_jsonl(analysis_manual_review_path, analysis_manual_review_rows)
    for row in all_rows:
        row.pop("retrieved_chunks_full", None)

    summary = {
        "analysis_type": "clean_dataset",
        "dataset": dataset,
        "configs": config_names,
        "logs_dir": str(logs_dir),
        "queries_path": str(queries_path),
        "summary_by_config": summary_by_config,
        "answer_quality_by_config": {
            config_name: config_summary.get("answer_quality", {})
            for config_name, config_summary in summary_by_config.items()
        },
        "metric_definitions": metric_definitions(),
        "reporting_notes": {
            "preferred_logs": "current_main_logs",
            "legacy_logs_policy": (
                "Older incomplete log schemas are not suitable for final thesis reporting."
            ),
            "clean_runs_role": (
                "Clean runs are the primary source for utility, benign refusal, and false-positive reporting."
            ),
        },
        "output_files": {
            "analysis_manual_review_path": str(analysis_manual_review_path),
            "config_summary_csv_path": str(config_summary_path),
            "latency_summary_csv_path": str(latency_summary_path),
        },
    }

    summary_path = output_dir / "summary.json"
    per_query_path = output_dir / "per_query_results.jsonl"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(per_query_path, all_rows)
    write_csv(
        config_summary_path,
        build_config_summary_rows(dataset=dataset, summary_by_config=summary_by_config),
        fieldnames=[
            "analysis_type",
            "dataset",
            "config_name",
            "stage_family",
            "n",
            "strict_rule_answer_match_count",
            "strict_rule_answer_match_n",
            "strict_rule_answer_match_rate",
            "lenient_match_count",
            "lenient_match_n",
            "lenient_match_rate",
            "validation_pass_count",
            "validation_pass_n",
            "validation_pass_rate",
            "benign_refusal_count",
            "benign_refusal_n",
            "benign_refusal_rate",
            "cannot_determine_count",
            "cannot_determine_n",
            "cannot_determine_rate",
            "final_output_block_or_replacement_count",
            "final_output_block_or_replacement_n",
            "final_output_block_or_replacement_rate",
            "retrieval_false_positive_runs_count",
            "retrieval_false_positive_runs_n",
            "retrieval_false_positive_runs_rate",
            "retrieval_flagged_benign_chunks_count",
            "retrieval_flagged_benign_chunks_n",
            "retrieval_flagged_benign_chunks_rate",
            "retrieval_downranked_benign_chunks_count",
            "retrieval_downranked_benign_chunks_n",
            "retrieval_downranked_benign_chunks_rate",
            "retrieval_removed_benign_top_k_chunks_count",
            "retrieval_removed_benign_top_k_chunks_n",
            "retrieval_removed_benign_top_k_chunks_rate",
            "prompt_false_positive_runs_count",
            "prompt_false_positive_runs_n",
            "prompt_false_positive_runs_rate",
            "benign_docs_labeled_low_trust_count",
            "benign_docs_labeled_low_trust_n",
            "benign_docs_labeled_low_trust_rate",
            "benign_docs_labeled_no_trust_count",
            "benign_docs_labeled_no_trust_n",
            "benign_docs_labeled_no_trust_rate",
            "benign_output_false_positive_count",
            "benign_output_false_positive_n",
            "benign_output_false_positive_rate",
            "answer_quality_refusal_count",
            "answer_quality_refusal_n",
            "answer_quality_refusal_rate",
            "mean_support_coverage",
            "mean_derived_total_latency_ms",
            "median_derived_total_latency_ms",
            "mean_non_generation_latency_ms",
            "median_non_generation_latency_ms",
            "mean_generation_latency_ms",
            "median_generation_latency_ms",
            "mean_retrieval_latency_ms",
            "mean_prompt_building_latency_ms",
            "mean_output_filtering_latency_ms",
            "mean_derived_total_latency_overhead_ms",
            "median_derived_total_latency_overhead_ms",
            "mean_derived_total_latency_overhead_pct",
            "median_derived_total_latency_overhead_pct",
            "mean_non_generation_latency_overhead_ms",
            "median_non_generation_latency_overhead_ms",
            "mean_non_generation_latency_overhead_pct",
            "median_non_generation_latency_overhead_pct",
        ],
    )
    write_csv(
        latency_summary_path,
        build_latency_summary_rows(dataset=dataset, summary_by_config=summary_by_config),
        fieldnames=[
            "analysis_type",
            "dataset",
            "config_name",
            "stage_family",
            "latency_scope",
            "latency_name",
            "count",
            "mean",
            "median",
            "min",
            "max",
            "std",
            "overhead_ms",
            "overhead_pct",
        ],
    )

    print(json.dumps(summary_by_config, indent=2, ensure_ascii=False))
    print(f"Saved summary -> {summary_path}")
    print(f"Saved per-query results -> {per_query_path}")
    print(f"Saved analysis manual review rows -> {analysis_manual_review_path}")
    print(f"Saved config summary CSV -> {config_summary_path}")
    print(f"Saved latency summary CSV -> {latency_summary_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Analyze clean dataset experiment logs.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--config",
        dest="config_names",
        action="append",
        default=None,
        help=(
            "Config name to analyze. Repeat the flag to analyze multiple configs, "
            "or pass --config all to analyze every config."
        ),
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Directory containing per-config experiment logs.",
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=None,
        help="Path to the query file used during the analyzed clean experiment runs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Base directory for analysis outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config_names = args.config_names or list(DEFAULT_CONFIGS)
    final_stage_dir = clean_processed_stage_dir(args.dataset, "final")
    analyze_clean_dataset(
        dataset=args.dataset,
        config_names=config_names,
        logs_dir=args.logs_dir or clean_logs_stage_dir(args.dataset, "final"),
        queries_path=args.queries_path or final_stage_dir / "queries.jsonl",
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
