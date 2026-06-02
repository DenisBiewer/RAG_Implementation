from __future__ import annotations

"""
Finalize reviewed PoisonedRAG target answers into a machine-consumable subset.

This mirrors the clean-query workflow:
- read the human-edited review file
- require explicit manual pass/fail decisions
- export the approved target-answer query rows
- write a manifest describing approvals and exclusions
"""

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .poisoned_rag_common import (
        DEFAULT_DATASET,
        DEFAULT_FINAL_TARGET_ANSWER_MANIFEST_PATH,
        DEFAULT_FINAL_TARGET_ANSWER_QUERIES_PATH,
        DEFAULT_TARGET_ANSWER_REVIEW_PATH,
        iter_jsonl,
        resolve_poisoned_rag_paths,
        write_jsonl,
    )
except ImportError:
    from poisoned_rag_common import (
        DEFAULT_DATASET,
        DEFAULT_FINAL_TARGET_ANSWER_MANIFEST_PATH,
        DEFAULT_FINAL_TARGET_ANSWER_QUERIES_PATH,
        DEFAULT_TARGET_ANSWER_REVIEW_PATH,
        iter_jsonl,
        resolve_poisoned_rag_paths,
        write_jsonl,
    )

from scripts.analysis.answer_validation import normalize_text


def load_manual_review_rows(path: Path) -> list[dict[str, Any]]:
    """Load the target-answer manual review artifact."""
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"No target-answer review rows found in {path}")
    return rows


def require_completed_manual_review(rows: list[dict[str, Any]]) -> None:
    """Require an explicit boolean pass/fail decision for every review row."""
    pending_query_ids = [
        row["query_id"]
        for row in rows
        if row.get("manual_target_answer_pass") not in {True, False}
    ]
    if pending_query_ids:
        raise ValueError(
            "Target-answer manual review is incomplete. Missing True/False "
            "manual_target_answer_pass for: "
            + ", ".join(pending_query_ids)
        )


def target_answer_is_usable(candidate: str, gold: str) -> bool:
    """Return whether one candidate target answer is non-empty and distinct from gold."""
    if not candidate.strip():
        return False
    return normalize_text(candidate) != normalize_text(gold)


def validate_review_row(row: dict[str, Any]) -> str | None:
    """
    Return a validation error code when the row is inconsistent.

    `manual_target_answer_pass=True` means the generated suggestion is accepted.
    `manual_target_answer_pass=False` means a manual override is required.
    """
    manual_pass = row["manual_target_answer_pass"]
    suggested_target_answer = (row.get("suggested_target_answer") or "").strip()
    manual_target_answer = (row.get("manual_target_answer") or "").strip()
    gold_answer = row["gold_answer"]

    if manual_pass is True and not target_answer_is_usable(suggested_target_answer, gold_answer):
        return "accepted_suggestion_is_not_usable"
    if manual_pass is False and not target_answer_is_usable(manual_target_answer, gold_answer):
        return "manual_override_missing_or_not_usable"
    return None


def build_final_target_answer_queries(
    manual_review_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Build finalized target-answer rows and count manual overrides."""
    require_completed_manual_review(manual_review_rows)

    final_rows: list[dict[str, Any]] = []
    manual_override_count = 0

    for row in manual_review_rows:
        validation_error = validate_review_row(row)
        if validation_error is not None:
            raise ValueError(
                f"Invalid target-answer review row for query_id={row['query_id']}: {validation_error}"
            )

        if row["manual_target_answer_pass"] is True:
            target_answer = row["suggested_target_answer"].strip()
            selection_method = "accepted_suggestion"
        else:
            target_answer = row["manual_target_answer"].strip()
            selection_method = "manual_override"
            manual_override_count += 1

        final_rows.append(
            {
                "query_id": row["query_id"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "target_answer": target_answer,
                "support_titles": row.get("support_titles", []),
                "target_answer_source": row.get("target_answer_source"),
                "target_answer_selection_method": selection_method,
                "suggested_target_answer": row.get("suggested_target_answer"),
                "manual_target_answer": row.get("manual_target_answer"),
                "answer_format_hint": row.get("answer_format_hint"),
                "review_notes": row.get("review_notes", ""),
            }
        )

    return final_rows, manual_override_count


def finalize_target_answers(
    *,
    manual_review_path: Path,
    final_queries_path: Path,
    final_manifest_path: Path,
) -> None:
    """Finalize approved target answers for downstream PoisonedRAG stages."""
    manual_review_rows = load_manual_review_rows(manual_review_path)
    final_queries, manual_override_count = build_final_target_answer_queries(manual_review_rows)

    write_jsonl(final_queries_path, final_queries)

    manifest = {
        "stage": "final",
        "script_name": Path(__file__).name,
        "manual_review_path": str(manual_review_path),
        "output_queries_path": str(final_queries_path),
        "manual_review_query_count": len(manual_review_rows),
        "final_query_count": len(final_queries),
        "accepted_suggestion_count": len(final_queries) - manual_override_count,
        "manual_override_count": manual_override_count,
        "final_query_ids": [row["query_id"] for row in final_queries],
    }
    final_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    final_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Loaded target-answer review rows: {len(manual_review_rows)}")
    print(f"Finalized target-answer queries: {len(final_queries)}")
    print(f"Manual overrides used: {manual_override_count}")
    print(f"Saved final target-answer queries -> {final_queries_path}")
    print(f"Saved manifest -> {final_manifest_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Finalize reviewed PoisonedRAG target answers."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--manual-review-path", type=Path, default=None)
    parser.add_argument("--final-queries-path", type=Path, default=None)
    parser.add_argument("--final-manifest-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    paths = resolve_poisoned_rag_paths(args.dataset)
    finalize_target_answers(
        manual_review_path=args.manual_review_path or paths["target_answer_review_path"],
        final_queries_path=args.final_queries_path or paths["final_target_answer_queries_path"],
        final_manifest_path=args.final_manifest_path or paths["final_target_answer_manifest_path"],
    )


if __name__ == "__main__":
    main()
