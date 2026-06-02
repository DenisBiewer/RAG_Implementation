from __future__ import annotations

"""
Finalize the reviewed close-ended target-query set.

Pipeline role:
- read the initial close-ended query/corpus stage
- read manual review decisions for script-validated queries
- require every reviewed query to have an explicit True/False decision
- if more queries passed manual review than needed, select them by seeded stable ranking
- write the final target queries, aligned final corpus, and a manifest
"""

import argparse
import json
import shutil
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.dataset_registry import (
    DEFAULT_CLOSE_ENDED_DATASET,
    clean_processed_stage_dir,
    get_close_ended_dataset_spec,
)
from app.io_utils import iter_jsonl, write_jsonl
from app.query_utils import unique_support_titles
from app.seed_config import resolve_seed
from app.selection_utils import select_seeded_subset

DEFAULT_DATASET = DEFAULT_CLOSE_ENDED_DATASET
DEFAULT_SPEC = get_close_ended_dataset_spec(DEFAULT_DATASET)
DEFAULT_INITIAL_STAGE_DIR = clean_processed_stage_dir(DEFAULT_DATASET, "initial")
DEFAULT_FINAL_STAGE_DIR = clean_processed_stage_dir(DEFAULT_DATASET, "final")

DEFAULT_INITIAL_QUERIES_PATH = DEFAULT_INITIAL_STAGE_DIR / "queries.jsonl"
DEFAULT_INITIAL_CORPUS_PATH = DEFAULT_INITIAL_STAGE_DIR / "corpus.jsonl"
DEFAULT_MANUAL_REVIEW_PATH = DEFAULT_FINAL_STAGE_DIR / "manual_review.jsonl"
DEFAULT_FINAL_QUERIES_PATH = DEFAULT_FINAL_STAGE_DIR / "queries.jsonl"
DEFAULT_FINAL_CORPUS_PATH = DEFAULT_FINAL_STAGE_DIR / "corpus.jsonl"
DEFAULT_FINAL_MANIFEST_PATH = DEFAULT_FINAL_STAGE_DIR / "manifest.json"

DEFAULT_FINAL_TARGET_COUNT = DEFAULT_SPEC.final_target_count
FINAL_SELECTION_NAMESPACE = "finalize_close_ended_query_set:manual_review_final_targets"

def load_manual_review_rows(path: Path) -> list[dict[str, Any]]:
    """Load the manual-review artifact."""
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"No manual review rows found in {path}")
    return rows


def require_completed_manual_review(rows: list[dict[str, Any]]) -> None:
    """Require an explicit boolean manual decision for every review row."""
    pending_query_ids = [
        row["query_id"]
        for row in rows
        if row.get("manual_validation_pass") not in {True, False}
    ]
    if pending_query_ids:
        raise ValueError(
            "Manual review is incomplete. Missing True/False manual_validation_pass for: "
            + ", ".join(pending_query_ids)
        )


def select_reviewed_rows(
    *,
    manual_review_rows: list[dict[str, Any]],
    final_target_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Select final reviewed rows, using seeded ranking only when needed."""
    require_completed_manual_review(manual_review_rows)

    accepted_rows = [
        row
        for row in manual_review_rows
        if row["manual_validation_pass"] is True
    ]
    if len(accepted_rows) < final_target_count:
        raise ValueError(
            f"Only {len(accepted_rows)} queries passed manual review, "
            f"but {final_target_count} final targets are required."
        )

    if len(accepted_rows) == final_target_count:
        return accepted_rows, [], False

    sampled_rows, not_selected_rows = select_seeded_subset(
        accepted_rows,
        count=final_target_count,
        seed=seed,
        id_fn=lambda row: row["query_id"],
        namespace=FINAL_SELECTION_NAMESPACE,
    )
    selected_ids = {row["query_id"] for row in sampled_rows}
    selected_rows = [
        row
        for row in manual_review_rows
        if row["query_id"] in selected_ids
    ]
    not_selected_rows = sorted(
        not_selected_rows,
        key=lambda row: str(row["query_id"]),
    )
    return selected_rows, not_selected_rows, True


def build_final_queries(
    *,
    initial_queries: list[dict[str, Any]],
    selected_review_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach acceptable answers and return final queries in source order."""
    query_by_id = {query["query_id"]: query for query in initial_queries}
    final_queries: list[dict[str, Any]] = []

    for row in selected_review_rows:
        source_query = query_by_id.get(row["query_id"])
        if source_query is None:
            raise ValueError(
                f"Manual review row references unknown query_id: {row['query_id']}"
            )
        final_query = dict(source_query)
        final_query["acceptable_answers"] = row.get("acceptable_answers") or [source_query["answer"]]
        final_queries.append(final_query)

    return final_queries


def build_final_corpus(
    *,
    initial_corpus: list[dict[str, Any]],
    initial_queries: list[dict[str, Any]],
    final_queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the final corpus from final-query support docs plus initial distractors."""
    initial_support_titles = {
        title
        for query in initial_queries
        for title in unique_support_titles(query)
    }

    ordered_final_support_titles: list[str] = []
    for query in final_queries:
        for title in unique_support_titles(query):
            if title not in ordered_final_support_titles:
                ordered_final_support_titles.append(title)

    support_docs_by_title: dict[str, dict[str, Any]] = {}
    distractors: list[dict[str, Any]] = []

    for record in initial_corpus:
        topic_id = record.get("topic_id")
        if topic_id in ordered_final_support_titles:
            support_docs_by_title.setdefault(topic_id, record)
        elif topic_id not in initial_support_titles:
            distractors.append(record)

    missing_titles = [
        title
        for title in ordered_final_support_titles
        if title not in support_docs_by_title
    ]
    if missing_titles:
        raise ValueError(
            "Initial corpus is missing final support docs: "
            + ", ".join(missing_titles)
        )

    final_corpus = [support_docs_by_title[title] for title in ordered_final_support_titles]
    final_corpus.extend(distractors)
    return final_corpus


def finalize_close_ended_query_set(
    *,
    dataset: str,
    initial_queries_path: Path,
    initial_corpus_path: Path,
    manual_review_path: Path,
    final_queries_path: Path,
    final_corpus_path: Path,
    final_manifest_path: Path,
    final_target_count: int,
    seed: int,
) -> None:
    """Finalize the reviewed target-query set and aligned final corpus."""
    dataset_spec = get_close_ended_dataset_spec(dataset)
    effective_seed = resolve_seed(seed)
    initial_queries = list(iter_jsonl(initial_queries_path))
    manual_review_rows = load_manual_review_rows(manual_review_path)

    selected_review_rows, not_selected_rows, used_random_sample = select_reviewed_rows(
        manual_review_rows=manual_review_rows,
        final_target_count=final_target_count,
        seed=effective_seed,
    )
    final_queries = build_final_queries(
        initial_queries=initial_queries,
        selected_review_rows=selected_review_rows,
    )

    write_jsonl(final_queries_path, final_queries)

    if dataset_spec.use_full_corpus:
        final_corpus_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(initial_corpus_path, final_corpus_path)
        final_corpus_record_count = sum(1 for _ in iter_jsonl(final_corpus_path))
        distractor_count = 0
    else:
        initial_corpus = list(iter_jsonl(initial_corpus_path))
        final_corpus = build_final_corpus(
            initial_corpus=initial_corpus,
            initial_queries=initial_queries,
            final_queries=final_queries,
        )
        write_jsonl(final_corpus_path, final_corpus)
        initial_support_titles = {
            title
            for query in initial_queries
            for title in unique_support_titles(query)
        }
        distractor_count = sum(
            1
            for record in final_corpus
            if record.get("topic_id") not in initial_support_titles
        )
        final_corpus_record_count = len(final_corpus)

    final_support_titles = {
        title
        for query in final_queries
        for title in unique_support_titles(query)
    }

    manifest = {
        "dataset_name": dataset,
        "stage": "final",
        "script_name": Path(__file__).name,
        "use_full_corpus": dataset_spec.use_full_corpus,
        "initial_queries_path": str(initial_queries_path),
        "initial_corpus_path": str(initial_corpus_path),
        "manual_review_path": str(manual_review_path),
        "output_queries_path": str(final_queries_path),
        "output_corpus_path": str(final_corpus_path),
        "final_target_count": final_target_count,
        "manual_review_query_count": len(manual_review_rows),
        "manual_pass_query_count": sum(
            row["manual_validation_pass"] is True for row in manual_review_rows
        ),
        "selected_final_query_count": len(final_queries),
        "used_random_sample": used_random_sample,
        "random_seed": effective_seed,
        "final_selection": {
            "method": (
                "stable_seeded_hash_rank"
                if used_random_sample
                else "all_manually_approved_rows_selected"
            ),
            "namespace": FINAL_SELECTION_NAMESPACE if used_random_sample else None,
            "candidate_count": sum(
                row["manual_validation_pass"] is True for row in manual_review_rows
            ),
            "selected_count": len(final_queries),
            "selected_query_ids": [query["query_id"] for query in final_queries],
            "not_selected_query_ids": [
                row["query_id"] for row in not_selected_rows
            ],
        },
        "support_doc_count": len(final_support_titles),
        "distractor_count": distractor_count,
        "final_corpus_record_count": final_corpus_record_count,
        "selected_query_ids": [query["query_id"] for query in final_queries],
        "manual_pass_not_selected_query_ids": [
            row["query_id"] for row in not_selected_rows
        ],
        "queries": [
            {
                "query_id": query["query_id"],
                "question": query["question"],
                "answer": query["answer"],
                "acceptable_answers": query["acceptable_answers"],
                "support_titles": unique_support_titles(query),
            }
            for query in final_queries
        ],
    }
    final_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    final_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    initial_corpus_record_count = sum(1 for _ in iter_jsonl(initial_corpus_path))
    print(f"Loaded initial queries: {len(initial_queries)}")
    print(f"Loaded initial corpus docs: {initial_corpus_record_count}")
    print(f"Loaded manual review rows: {len(manual_review_rows)}")
    print(f"Selected final target queries: {len(final_queries)}")
    print(f"Used random sample: {used_random_sample}")
    print(f"Saved final queries -> {final_queries_path}")
    print(f"Saved final corpus -> {final_corpus_path}")
    print(f"Saved final manifest -> {final_manifest_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Finalize the reviewed close-ended target-query set."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--initial-queries-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--initial-corpus-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--manual-review-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--final-queries-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--final-corpus-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--final-manifest-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--final-target-count",
        type=int,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    dataset_spec = get_close_ended_dataset_spec(args.dataset)
    initial_stage_dir = clean_processed_stage_dir(args.dataset, "initial")
    final_stage_dir = clean_processed_stage_dir(args.dataset, "final")
    finalize_close_ended_query_set(
        dataset=args.dataset,
        initial_queries_path=args.initial_queries_path or initial_stage_dir / "queries.jsonl",
        initial_corpus_path=args.initial_corpus_path or initial_stage_dir / "corpus.jsonl",
        manual_review_path=args.manual_review_path or final_stage_dir / "manual_review.jsonl",
        final_queries_path=args.final_queries_path or final_stage_dir / "queries.jsonl",
        final_corpus_path=args.final_corpus_path or final_stage_dir / "corpus.jsonl",
        final_manifest_path=args.final_manifest_path or final_stage_dir / "manifest.json",
        final_target_count=args.final_target_count or dataset_spec.final_target_count,
        seed=resolve_seed(args.seed),
    )


if __name__ == "__main__":
    main()
