from __future__ import annotations

"""
Analyze clean validation correctness logs for one or more experiment configs.

Outputs:
- results/validation/<dataset>/summary.json
- results/validation/<dataset>/per_query_results.jsonl
- data/processed/clean/<dataset>/script_validated/validated_queries.jsonl
- data/processed/clean/<dataset>/script_validated/failed_validation_queries.jsonl
- data/processed/clean/<dataset>/script_validated/validation_manifest.json
- data/processed/clean/<dataset>/final/manual_review.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any

from answer_validation import (
    _boolean_label_from_segment,
    _final_answer_segments,
    _matches_boolean_answer,
    _matches_span_answer,
    answer_matches,
    contains_rejection_phrase,
    normalize_text,
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = next(
    parent
    for parent in (BASE_DIR, *BASE_DIR.parents)
    if (parent / "app").exists() and (parent / "data").exists()
)
LOG_DIR = PROJECT_DIR / "logs"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

DEFAULT_DATASET = "close_ended_dev"
CLOSE_ENDED_DEV_DIR = PROCESSED_DIR / "clean" / DEFAULT_DATASET
INITIAL_STAGE_DIR = CLOSE_ENDED_DEV_DIR / "initial"
SCRIPT_VALIDATED_STAGE_DIR = CLOSE_ENDED_DEV_DIR / "script_validated"
FINAL_STAGE_DIR = CLOSE_ENDED_DEV_DIR / "final"
ALL_CONFIG_NAMES = [
    "baseline",
    "retrieval_only",
    "prompt_only",
    "output_only",
]
DEFAULT_CONFIGS = ["all"]


def iter_jsonl(path: Path):
    """Yield JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSON objects to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def fraction(count: int | float, total: int | float) -> float:
    """Return a rounded fraction with zero-safe behavior."""
    if total == 0:
        return 0.0
    return round(count / total, 4)


def unique_support_titles(query_record: dict[str, Any]) -> list[str]:
    """Return unique supporting titles while preserving order."""
    return list(dict.fromkeys(query_record["supporting_facts"]["title"]))


def compact_retrieved_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Keep only retrieval metadata needed for validation analysis."""
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "topic_id": chunk["topic_id"],
        "rank": chunk["rank"],
        "score": round(chunk["score"], 6),
        "is_malicious": chunk["is_malicious"],
        "attack_type": chunk.get("attack_type"),
    }




def build_support_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Extract compact support-coverage diagnostics from a run record."""
    coverage = record.get("support_coverage", {})
    titles_in_corpus = coverage.get("titles_in_corpus", {})
    titles_retrieved = coverage.get("titles_retrieved", {})

    return {
        "titles_in_corpus": titles_in_corpus,
        "titles_retrieved": titles_retrieved,
        "all_support_titles_present_in_corpus": bool(titles_in_corpus) and all(titles_in_corpus.values()),
        "all_support_titles_retrieved": bool(titles_retrieved) and all(titles_retrieved.values()),
    }


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw experiment record into compact validation-analysis data."""
    expected_answer = record["expected_answer"]
    final_output = record["final_output"]
    strict_match, lenient_match = answer_matches(final_output, expected_answer)
    support_summary = build_support_summary(record)
    strict_rule_answer_match = strict_match
    validation_pass = strict_rule_answer_match and support_summary["all_support_titles_retrieved"]

    return {
        "query_id": record["query_id"],
        "config_name": record["config_name"],
        "question": record["query"],
        "expected_answer": expected_answer,
        "expected_answer_normalized": normalize_text(expected_answer),
        "answer_type": "boolean" if normalize_text(expected_answer) in {"yes", "no"} else "span",
        "contains_rejection_phrase": contains_rejection_phrase(final_output),
        "strict_rule_answer_match": strict_rule_answer_match,
        "strict_match": strict_match,
        "lenient_match": lenient_match,
        "validation_pass": validation_pass,
        "support_titles": record["supporting_facts"]["title"],
        "support_titles_present_in_corpus": support_summary["titles_in_corpus"],
        "support_titles_retrieved": support_summary["titles_retrieved"],
        "all_support_titles_present_in_corpus": support_summary["all_support_titles_present_in_corpus"],
        "all_support_titles_retrieved": support_summary["all_support_titles_retrieved"],
        "retrieved_chunks_compact": [
            compact_retrieved_chunk(chunk)
            for chunk in record["retrieved_chunks"]
        ],
        "model_output_compact": record["model_output"],
        "final_output_compact": final_output,
        "defenses_enabled": record.get("defense_decisions", {}).get("defenses_enabled", []),
        "latencies_ms": record.get("latencies_ms", {}),
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

    boolean_results = [result for result in results if result["answer_type"] == "boolean"]
    span_results = [result for result in results if result["answer_type"] == "span"]

    return {
        "total_records": total,
        "strict_rule_answer_matches": strict_matches,
        "strict_rule_answer_match_rate": fraction(strict_matches, total),
        "lenient_matches": lenient_matches,
        "lenient_match_rate": fraction(lenient_matches, total),
        "all_support_titles_present_in_corpus": support_present,
        "all_support_titles_present_in_corpus_rate": fraction(support_present, total),
        "all_support_titles_retrieved": support_retrieved,
        "all_support_titles_retrieved_rate": fraction(support_retrieved, total),
        "validation_passes": validation_passes,
        "validation_pass_rate": fraction(validation_passes, total),
        "rejection_phrase_outputs": rejection_phrase_outputs,
        "rejection_phrase_output_rate": fraction(rejection_phrase_outputs, total),
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
    }


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

    for record in iter_jsonl(log_path):
        if record["query_id"] not in known_query_ids:
            skipped += 1
            continue
        rows.append(evaluate_record(record))

    summary = summarize_config(rows)
    summary["status"] = "ok"
    summary["log_path"] = str(log_path)
    summary["skipped_records_without_query"] = skipped
    summary["config_name"] = config_name
    return rows, summary


def export_query_subsets(
    *,
    dataset: str,
    source_queries: list[dict[str, Any]],
    results_by_query_id: dict[str, dict[str, Any]],
    validated_queries_path: Path,
    failed_queries_path: Path,
    manifest_path: Path,
    export_config_name: str,
) -> dict[str, Any]:
    """Write the queries that passed validation and those that did not."""
    validated_queries: list[dict[str, Any]] = []
    failed_queries: list[dict[str, Any]] = []
    missing_query_ids: list[str] = []

    for query in source_queries:
        result = results_by_query_id.get(query["query_id"])
        if result is None:
            missing_query_ids.append(query["query_id"])
            continue
        if result["validation_pass"]:
            validated_queries.append(query)
        else:
            failed_queries.append(query)

    write_jsonl(validated_queries_path, validated_queries)
    write_jsonl(failed_queries_path, failed_queries)

    manifest = {
        "dataset": dataset,
        "export_config_name": export_config_name,
        "source_query_count": len(source_queries),
        "validated_query_count": len(validated_queries),
        "failed_query_count": len(failed_queries),
        "missing_from_log_query_count": len(missing_query_ids),
        "missing_from_log_query_ids": missing_query_ids,
        "validated_queries_path": str(validated_queries_path),
        "failed_queries_path": str(failed_queries_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_existing_manual_review_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Load existing manual-review rows keyed by query id when present."""
    if not path.exists():
        return {}
    return {
        row["query_id"]: row
        for row in iter_jsonl(path)
    }


def build_manual_review_rows(
    validated_queries: list[dict[str, Any]],
    results_by_query_id: dict[str, dict[str, Any]],
    existing_rows_by_query_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build or refresh the manual-review artifact for script-validated queries."""
    rows: list[dict[str, Any]] = []

    for query in validated_queries:
        query_id = query["query_id"]
        result = results_by_query_id[query_id]
        existing = existing_rows_by_query_id.get(query_id, {})
        acceptable_answers = existing.get("acceptable_answers") or [query["answer"]]

        rows.append(
            {
                "query_id": query_id,
                "question": query["question"],
                "expected_answer": query["answer"],
                "actual_answer": result["final_output_compact"],
                "manual_validation_pass": existing.get("manual_validation_pass"),
                "support_titles": unique_support_titles(query),
                "script_validation_pass": result["validation_pass"],
                "strict_rule_answer_match": result["strict_rule_answer_match"],
                "all_support_titles_retrieved": result["all_support_titles_retrieved"],
                "reason": existing.get("reason", ""),
                "acceptable_answers": acceptable_answers,
            }
        )

    return rows


def analyze_clean_baseline_validation(
    *,
    dataset: str,
    config_names: list[str],
    export_config_name: str,
    logs_dir: Path,
    queries_path: Path,
    results_dir: Path,
    validated_queries_path: Path,
    failed_queries_path: Path,
    export_manifest_path: Path,
    manual_review_path: Path,
) -> None:
    """Analyze selected clean validation logs and export passed queries."""
    config_names = resolve_config_names(config_names)
    source_queries = load_queries(queries_path)
    known_query_ids = {query["query_id"] for query in source_queries}

    all_rows: list[dict[str, Any]] = []
    rows_by_config: dict[str, list[dict[str, Any]]] = {}
    summary_by_config: dict[str, dict[str, Any]] = {}

    for config_name in config_names:
        log_path = logs_dir / f"{config_name}_runs.jsonl"
        rows, summary = analyze_config_log(
            config_name=config_name,
            log_path=log_path,
            known_query_ids=known_query_ids,
        )
        all_rows.extend(rows)
        rows_by_config[config_name] = rows
        summary_by_config[config_name] = summary

    if export_config_name == "all":
        raise ValueError("Export config must be one specific config name, not 'all'.")

    if export_config_name not in rows_by_config:
        raise ValueError(
            f"Export config '{export_config_name}' is not in the analyzed config list."
        )

    export_results_by_query_id = {
        row["query_id"]: row
        for row in rows_by_config[export_config_name]
    }
    export_manifest = export_query_subsets(
        dataset=dataset,
        source_queries=source_queries,
        results_by_query_id=export_results_by_query_id,
        validated_queries_path=validated_queries_path,
        failed_queries_path=failed_queries_path,
        manifest_path=export_manifest_path,
        export_config_name=export_config_name,
    )
    validated_queries = [
        query
        for query in source_queries
        if export_results_by_query_id.get(query["query_id"], {}).get("validation_pass")
    ]
    existing_manual_rows = load_existing_manual_review_rows(manual_review_path)
    manual_review_rows = build_manual_review_rows(
        validated_queries=validated_queries,
        results_by_query_id=export_results_by_query_id,
        existing_rows_by_query_id=existing_manual_rows,
    )
    write_jsonl(manual_review_path, manual_review_rows)

    output_dir = results_dir / "validation" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "analysis_type": "clean_baseline_validation",
        "dataset": dataset,
        "configs": config_names,
        "export_config_name": export_config_name,
        "logs_dir": str(logs_dir),
        "queries_path": str(queries_path),
        "summary_by_config": summary_by_config,
        "export_manifest": export_manifest,
        "manual_review_path": str(manual_review_path),
    }

    summary_path = output_dir / "summary.json"
    per_query_path = output_dir / "per_query_results.jsonl"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(per_query_path, all_rows)

    print(json.dumps(summary_by_config, indent=2, ensure_ascii=False))
    print(f"Saved summary -> {summary_path}")
    print(f"Saved per-query results -> {per_query_path}")
    print(f"Saved validated queries -> {validated_queries_path}")
    print(f"Saved failed queries -> {failed_queries_path}")
    print(f"Saved validation manifest -> {export_manifest_path}")
    print(f"Saved manual review artifact -> {manual_review_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Analyze clean validation logs.")
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
        "--export-config",
        default="baseline",
        help="Specific config whose passed queries should be exported for later use.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=LOG_DIR / "clean" / DEFAULT_DATASET / "initial",
        help="Directory containing per-config experiment logs.",
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=INITIAL_STAGE_DIR / "queries.jsonl",
        help="Path to the source query file used during clean baseline runs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Base directory for analysis outputs.",
    )
    parser.add_argument(
        "--validated-queries-path",
        type=Path,
        default=SCRIPT_VALIDATED_STAGE_DIR / "validated_queries.jsonl",
        help="Path to the query subset that passed clean baseline validation.",
    )
    parser.add_argument(
        "--failed-queries-path",
        type=Path,
        default=SCRIPT_VALIDATED_STAGE_DIR / "failed_validation_queries.jsonl",
        help="Path to the query subset that failed clean baseline validation.",
    )
    parser.add_argument(
        "--export-manifest-path",
        type=Path,
        default=SCRIPT_VALIDATED_STAGE_DIR / "validation_manifest.json",
        help="Path to the manifest describing the exported validation subsets.",
    )
    parser.add_argument(
        "--manual-review-path",
        type=Path,
        default=FINAL_STAGE_DIR / "manual_review.jsonl",
        help="Path to the manual review artifact for script-validated queries.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config_names = args.config_names or list(DEFAULT_CONFIGS)
    analyze_clean_baseline_validation(
        dataset=args.dataset,
        config_names=config_names,
        export_config_name=args.export_config,
        logs_dir=args.logs_dir,
        queries_path=args.queries_path,
        results_dir=args.results_dir,
        validated_queries_path=args.validated_queries_path,
        failed_queries_path=args.failed_queries_path,
        export_manifest_path=args.export_manifest_path,
        manual_review_path=args.manual_review_path,
    )


if __name__ == "__main__":
    main()
