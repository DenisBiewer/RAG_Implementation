from __future__ import annotations

"""
Build a controlled HotpotQA development subset with complete support coverage.

The original `make_dev_subset.py` takes the first N corpus documents and first
N queries independently. That is fast, but it can create query/corpus pairs
where the supporting documents for a query are missing from the dev corpus.

This script does the opposite:
- choose queries whose supporting titles exist in the full processed corpus
- include every supporting document for those queries
- add random distractor documents so retrieval still has non-gold candidates

The resulting files are intended for attack/defense smoke tests, not final
thesis-scale evaluation.
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

CLEAN_COMPLETE_DIR = PROCESSED_DIR / "clean" / "complete"
CLEAN_CONTROLLED_DEV_DIR = PROCESSED_DIR / "clean" / "controlled_dev"
CLEAN_DEV_DIR = PROCESSED_DIR / "clean" / "dev"

DEFAULT_CORPUS_PATH = CLEAN_COMPLETE_DIR / "corpus.jsonl"
DEFAULT_QUERIES_PATH = CLEAN_COMPLETE_DIR / "queries.jsonl"
DEFAULT_OUTPUT_CORPUS_PATH = CLEAN_CONTROLLED_DEV_DIR / "corpus.jsonl"
DEFAULT_OUTPUT_QUERIES_PATH = CLEAN_CONTROLLED_DEV_DIR / "queries.jsonl"
DEFAULT_MANIFEST_PATH = CLEAN_CONTROLLED_DEV_DIR / "manifest.json"

STANDARD_DEV_CORPUS_PATH = CLEAN_DEV_DIR / "corpus.jsonl"
STANDARD_DEV_QUERIES_PATH = CLEAN_DEV_DIR / "queries.jsonl"


def iter_jsonl(path: Path):
    """Yield JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_corpus_titles(corpus_path: Path) -> set[str]:
    """
    Collect all topic ids available in the full processed corpus.

    The corpus title is stored as `topic_id`, which matches HotpotQA supporting
    fact titles in the processed query file.
    """
    return {
        record["topic_id"]
        for record in iter_jsonl(corpus_path)
        if record.get("topic_id")
    }


def unique_support_titles(query_record: dict[str, Any]) -> list[str]:
    """Return unique supporting titles while preserving their original order."""
    titles = query_record["supporting_facts"]["title"]
    return list(dict.fromkeys(titles))


def select_queries(
    queries_path: Path,
    corpus_titles: set[str],
    query_count: int,
) -> list[dict[str, Any]]:
    """Select the first `query_count` queries whose support docs all exist."""
    selected: list[dict[str, Any]] = []

    for query in iter_jsonl(queries_path):
        support_titles = unique_support_titles(query)
        if support_titles and all(title in corpus_titles for title in support_titles):
            selected.append(query)

        if len(selected) >= query_count:
            break

    if len(selected) < query_count:
        raise ValueError(
            f"Only found {len(selected)} queries with complete support coverage; "
            f"requested {query_count}."
        )

    return selected


def collect_support_docs_and_distractors(
    corpus_path: Path,
    support_titles: set[str],
    distractor_count: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """
    Collect required support documents and reservoir-sample distractors.

    Reservoir sampling avoids loading the full 300MB+ corpus into memory.
    """
    rng = random.Random(seed)
    support_docs: dict[str, dict[str, Any]] = {}
    distractors: list[dict[str, Any]] = []
    seen_distractors = 0

    for record in iter_jsonl(corpus_path):
        topic_id = record.get("topic_id")
        if topic_id in support_titles:
            support_docs.setdefault(topic_id, record)
            continue

        seen_distractors += 1
        if len(distractors) < distractor_count:
            distractors.append(record)
            continue

        replacement_index = rng.randrange(seen_distractors)
        if replacement_index < distractor_count:
            distractors[replacement_index] = record

    missing = sorted(support_titles - set(support_docs))
    if missing:
        raise ValueError(f"Missing support docs after corpus scan: {missing}")

    return support_docs, distractors


def build_controlled_dev_set(
    *,
    corpus_path: Path,
    queries_path: Path,
    output_corpus_path: Path,
    output_queries_path: Path,
    manifest_path: Path,
    query_count: int,
    distractor_count: int,
    seed: int,
    replace_standard_dev: bool,
) -> None:
    """Build and write the controlled dev corpus, query file, and manifest."""
    print(f"Loading corpus title set from {corpus_path}...")
    corpus_titles = collect_corpus_titles(corpus_path)
    print(f"Found {len(corpus_titles)} unique corpus titles.")

    print(f"Selecting {query_count} queries with complete support coverage...")
    selected_queries = select_queries(queries_path, corpus_titles, query_count)

    ordered_support_titles: list[str] = []
    for query in selected_queries:
        for title in unique_support_titles(query):
            if title not in ordered_support_titles:
                ordered_support_titles.append(title)

    support_title_set = set(ordered_support_titles)
    print(f"Selected {len(selected_queries)} queries with {len(support_title_set)} unique support docs.")

    print(f"Collecting support docs and {distractor_count} random distractors...")
    support_docs, distractors = collect_support_docs_and_distractors(
        corpus_path=corpus_path,
        support_titles=support_title_set,
        distractor_count=distractor_count,
        seed=seed,
    )

    controlled_corpus = [support_docs[title] for title in ordered_support_titles]
    controlled_corpus.extend(distractors)

    write_jsonl(output_queries_path, selected_queries)
    write_jsonl(output_corpus_path, controlled_corpus)

    manifest = {
        "query_count": len(selected_queries),
        "support_doc_count": len(ordered_support_titles),
        "distractor_count": len(distractors),
        "corpus_doc_count": len(controlled_corpus),
        "seed": seed,
        "source_corpus_path": str(corpus_path),
        "source_queries_path": str(queries_path),
        "output_corpus_path": str(output_corpus_path),
        "output_queries_path": str(output_queries_path),
        "queries": [
            {
                "query_id": query["query_id"],
                "question": query["question"],
                "answer": query["answer"],
                "support_titles": unique_support_titles(query),
            }
            for query in selected_queries
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if replace_standard_dev:
        write_jsonl(STANDARD_DEV_QUERIES_PATH, selected_queries)
        write_jsonl(STANDARD_DEV_CORPUS_PATH, controlled_corpus)
        print(f"Also replaced clean dev queries: {STANDARD_DEV_QUERIES_PATH}")
        print(f"Also replaced clean dev corpus: {STANDARD_DEV_CORPUS_PATH}")

    print(f"Saved controlled dev queries: {len(selected_queries)} -> {output_queries_path}")
    print(f"Saved controlled dev corpus: {len(controlled_corpus)} -> {output_corpus_path}")
    print(f"Saved manifest -> {manifest_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Build a controlled dev set with complete support documents."
    )
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--queries-path", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--output-corpus-path", type=Path, default=DEFAULT_OUTPUT_CORPUS_PATH)
    parser.add_argument("--output-queries-path", type=Path, default=DEFAULT_OUTPUT_QUERIES_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--query-count", type=int, default=10)
    parser.add_argument("--distractor-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--replace-standard-dev",
        action="store_true",
        help="Also overwrite data/processed/clean/dev/corpus.jsonl and queries.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    build_controlled_dev_set(
        corpus_path=args.corpus_path,
        queries_path=args.queries_path,
        output_corpus_path=args.output_corpus_path,
        output_queries_path=args.output_queries_path,
        manifest_path=args.manifest_path,
        query_count=args.query_count,
        distractor_count=args.distractor_count,
        seed=args.seed,
        replace_standard_dev=args.replace_standard_dev,
    )


if __name__ == "__main__":
    main()
