from __future__ import annotations

"""
Prepare close-ended dataset finalization artifacts from clean validation logs.

Outputs:
- data/processed/clean/<dataset>/script_validated/validated_queries.jsonl
- data/processed/clean/<dataset>/script_validated/failed_validation_queries.jsonl
- data/processed/clean/<dataset>/script_validated/validation_manifest.json
- data/processed/clean/<dataset>/final/manual_review.jsonl
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.dataset_registry import (
    DEFAULT_CLOSE_ENDED_DATASET,
    clean_logs_stage_dir,
    clean_processed_stage_dir,
)
from app.io_utils import iter_jsonl, write_jsonl
from app.query_utils import unique_support_titles
from scripts.analysis.analyze_clean_dataset import (
    analyze_config_log,
    load_queries,
    resolve_config_names,
)

DEFAULT_DATASET = DEFAULT_CLOSE_ENDED_DATASET
DEFAULT_CONFIG_NAME = "baseline"


def export_query_subsets(
    *,
    dataset: str,
    source_queries: list[dict[str, Any]],
    results_by_query_id: dict[str, dict[str, Any]],
    validated_queries_path: Path,
    failed_queries_path: Path,
    manifest_path: Path,
    config_name: str,
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
        "config_name": config_name,
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


def validate_close_ended_query_set(
    *,
    dataset: str,
    config_name: str,
    logs_dir: Path,
    queries_path: Path,
    validated_queries_path: Path,
    failed_queries_path: Path,
    export_manifest_path: Path,
    manual_review_path: Path,
) -> None:
    """Prepare script-validated subsets and manual-review rows for finalization."""
    config_name = resolve_config_names([config_name])[0]
    source_queries = load_queries(queries_path)
    known_query_ids = {query["query_id"] for query in source_queries}
    log_path = logs_dir / f"{config_name}_runs.jsonl"
    rows, summary = analyze_config_log(
        config_name=config_name,
        log_path=log_path,
        known_query_ids=known_query_ids,
    )
    if summary["status"] != "ok":
        raise ValueError(f"Required validation log is missing or unreadable: {log_path}")

    results_by_query_id = {
        row["query_id"]: row
        for row in rows
    }
    export_manifest = export_query_subsets(
        dataset=dataset,
        source_queries=source_queries,
        results_by_query_id=results_by_query_id,
        validated_queries_path=validated_queries_path,
        failed_queries_path=failed_queries_path,
        manifest_path=export_manifest_path,
        config_name=config_name,
    )
    validated_queries = [
        query
        for query in source_queries
        if results_by_query_id.get(query["query_id"], {}).get("validation_pass")
    ]
    existing_manual_rows = load_existing_manual_review_rows(manual_review_path)
    manual_review_rows = build_manual_review_rows(
        validated_queries=validated_queries,
        results_by_query_id=results_by_query_id,
        existing_rows_by_query_id=existing_manual_rows,
    )
    write_jsonl(manual_review_path, manual_review_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved validated queries -> {validated_queries_path}")
    print(f"Saved failed queries -> {failed_queries_path}")
    print(f"Saved validation manifest -> {export_manifest_path}")
    print(f"Saved manual review artifact -> {manual_review_path}")
    print(
        "Prepared finalization inputs: "
        f"{export_manifest['validated_query_count']} validated, "
        f"{export_manifest['failed_query_count']} failed."
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Validate the close-ended query set from clean validation logs."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--config",
        dest="config_name",
        default=DEFAULT_CONFIG_NAME,
        help="Config name whose clean validation log should be used for dataset preparation.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Directory containing the clean validation logs for the initial dataset stage.",
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=None,
        help="Path to the initial close-ended query file used during clean validation.",
    )
    parser.add_argument(
        "--validated-queries-path",
        type=Path,
        default=None,
        help="Path to the query subset that passed clean baseline validation.",
    )
    parser.add_argument(
        "--failed-queries-path",
        type=Path,
        default=None,
        help="Path to the query subset that failed clean baseline validation.",
    )
    parser.add_argument(
        "--export-manifest-path",
        type=Path,
        default=None,
        help="Path to the manifest describing the exported validation subsets.",
    )
    parser.add_argument(
        "--manual-review-path",
        type=Path,
        default=None,
        help="Path to the manual review artifact consumed by finalization.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    initial_stage_dir = clean_processed_stage_dir(args.dataset, "initial")
    script_validated_stage_dir = clean_processed_stage_dir(args.dataset, "script_validated")
    final_stage_dir = clean_processed_stage_dir(args.dataset, "final")
    validate_close_ended_query_set(
        dataset=args.dataset,
        config_name=args.config_name,
        logs_dir=args.logs_dir or clean_logs_stage_dir(args.dataset, "initial"),
        queries_path=args.queries_path or initial_stage_dir / "queries.jsonl",
        validated_queries_path=args.validated_queries_path or script_validated_stage_dir / "validated_queries.jsonl",
        failed_queries_path=args.failed_queries_path or script_validated_stage_dir / "failed_validation_queries.jsonl",
        export_manifest_path=args.export_manifest_path or script_validated_stage_dir / "validation_manifest.json",
        manual_review_path=args.manual_review_path or final_stage_dir / "manual_review.jsonl",
    )


if __name__ == "__main__":
    main()
