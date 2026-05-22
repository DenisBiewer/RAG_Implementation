from __future__ import annotations

"""
Finalize the reviewed close-ended target-query set.

Pipeline role:
- read the initial close-ended query/corpus stage
- read manual review decisions for script-validated queries
- require every reviewed query to have an explicit True/False decision
- if more queries passed manual review than needed, draw a seeded random sample
- write the final target queries, aligned final corpus, and a manifest
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = next(
    parent
    for parent in (BASE_DIR, *BASE_DIR.parents)
    if (parent / "app").exists() and (parent / "data").exists()
)
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

DEFAULT_DATASET = "close_ended_dev"
CLOSE_ENDED_DEV_DIR = PROCESSED_DIR / "clean" / DEFAULT_DATASET
INITIAL_STAGE_DIR = CLOSE_ENDED_DEV_DIR / "initial"
FINAL_STAGE_DIR = CLOSE_ENDED_DEV_DIR / "final"

DEFAULT_INITIAL_QUERIES_PATH = INITIAL_STAGE_DIR / "queries.jsonl"
DEFAULT_INITIAL_CORPUS_PATH = INITIAL_STAGE_DIR / "corpus.jsonl"
DEFAULT_MANUAL_REVIEW_PATH = FINAL_STAGE_DIR / "manual_review.jsonl"
DEFAULT_FINAL_QUERIES_PATH = FINAL_STAGE_DIR / "queries.jsonl"
DEFAULT_FINAL_CORPUS_PATH = FINAL_STAGE_DIR / "corpus.jsonl"
DEFAULT_FINAL_MANIFEST_PATH = FINAL_STAGE_DIR / "manifest.json"

DEFAULT_FINAL_TARGET_COUNT = 10
DEFAULT_SEED = 42


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


def unique_support_titles(query_record: dict[str, Any]) -> list[str]:
    """Return unique supporting titles while preserving order."""
    return list(dict.fromkeys(query_record["supporting_facts"]["title"]))


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
    """Select final reviewed rows, using seeded sampling only when needed."""
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

    rng = random.Random(seed)
    sampled_rows = rng.sample(accepted_rows, final_target_count)
    selected_ids = {row["query_id"] for row in sampled_rows}
    selected_rows = [
        row
        for row in manual_review_rows
        if row["query_id"] in selected_ids
    ]
    not_selected_rows = [
        row
        for row in accepted_rows
        if row["query_id"] not in selected_ids
    ]
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
    initial_queries = list(iter_jsonl(initial_queries_path))
    initial_corpus = list(iter_jsonl(initial_corpus_path))
    manual_review_rows = load_manual_review_rows(manual_review_path)

    selected_review_rows, not_selected_rows, used_random_sample = select_reviewed_rows(
        manual_review_rows=manual_review_rows,
        final_target_count=final_target_count,
        seed=seed,
    )
    final_queries = build_final_queries(
        initial_queries=initial_queries,
        selected_review_rows=selected_review_rows,
    )
    final_corpus = build_final_corpus(
        initial_corpus=initial_corpus,
        initial_queries=initial_queries,
        final_queries=final_queries,
    )

    write_jsonl(final_queries_path, final_queries)
    write_jsonl(final_corpus_path, final_corpus)

    final_support_titles = {
        title
        for query in final_queries
        for title in unique_support_titles(query)
    }
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

    manifest = {
        "dataset_name": dataset,
        "stage": "final",
        "script_name": Path(__file__).name,
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
        "random_seed": seed,
        "support_doc_count": len(final_support_titles),
        "distractor_count": distractor_count,
        "final_corpus_record_count": len(final_corpus),
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

    print(f"Loaded initial queries: {len(initial_queries)}")
    print(f"Loaded initial corpus docs: {len(initial_corpus)}")
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
        default=DEFAULT_INITIAL_QUERIES_PATH,
    )
    parser.add_argument(
        "--initial-corpus-path",
        type=Path,
        default=DEFAULT_INITIAL_CORPUS_PATH,
    )
    parser.add_argument(
        "--manual-review-path",
        type=Path,
        default=DEFAULT_MANUAL_REVIEW_PATH,
    )
    parser.add_argument(
        "--final-queries-path",
        type=Path,
        default=DEFAULT_FINAL_QUERIES_PATH,
    )
    parser.add_argument(
        "--final-corpus-path",
        type=Path,
        default=DEFAULT_FINAL_CORPUS_PATH,
    )
    parser.add_argument(
        "--final-manifest-path",
        type=Path,
        default=DEFAULT_FINAL_MANIFEST_PATH,
    )
    parser.add_argument(
        "--final-target-count",
        type=int,
        default=DEFAULT_FINAL_TARGET_COUNT,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    finalize_close_ended_query_set(
        dataset=args.dataset,
        initial_queries_path=args.initial_queries_path,
        initial_corpus_path=args.initial_corpus_path,
        manual_review_path=args.manual_review_path,
        final_queries_path=args.final_queries_path,
        final_corpus_path=args.final_corpus_path,
        final_manifest_path=args.final_manifest_path,
        final_target_count=args.final_target_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
