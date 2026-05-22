from __future__ import annotations

"""
Build a small clean HotpotQA development subset of close-ended questions.

This script follows the existing dataset-builder conventions in the project:
- read clean source corpus and queries from `data/processed/clean/complete`
- conservatively filter queries for close-ended, substring-checkable answers
- require local corpus support coverage
- build a small aligned clean corpus with support documents plus distractors
- write the selected query set, corpus, audit files, and manifest

The output is intended as a reusable clean development set for later attack or
defense experiments. No malicious or poisoned documents are added.
"""

import argparse
import json
import random
import re
from collections import Counter
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
CLOSE_ENDED_DEV_DIR = PROCESSED_DIR / "clean" / "close_ended_dev"
INITIAL_OUTPUT_DIR = CLOSE_ENDED_DEV_DIR / "initial"

DEFAULT_CORPUS_PATH = CLEAN_COMPLETE_DIR / "corpus.jsonl"
DEFAULT_QUERIES_PATH = CLEAN_COMPLETE_DIR / "queries.jsonl"
DEFAULT_OUTPUT_CORPUS_PATH = INITIAL_OUTPUT_DIR / "corpus.jsonl"
DEFAULT_OUTPUT_QUERIES_PATH = INITIAL_OUTPUT_DIR / "queries.jsonl"
DEFAULT_ACCEPTED_AUDIT_PATH = INITIAL_OUTPUT_DIR / "accepted_candidates.jsonl"
DEFAULT_REJECTED_AUDIT_PATH = INITIAL_OUTPUT_DIR / "rejected_candidates.jsonl"
DEFAULT_MANIFEST_PATH = INITIAL_OUTPUT_DIR / "manifest.json"

DEFAULT_INITIAL_CANDIDATE_COUNT = 50
DEFAULT_FINAL_TARGET_COUNT = 10
DEFAULT_DISTRACTOR_COUNT = 500
DEFAULT_SEED = 42
MAX_SHORT_ANSWER_WORDS = 5
MAX_ALLOWED_COMMAS = 1

MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

BOOLEAN_QUESTION_PREFIXES = (
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "has ",
    "have ",
    "had ",
    "can ",
    "could ",
    "will ",
    "would ",
)

PREFERRED_QUESTION_PREFIXES = (
    "who ",
    "what ",
    "where ",
    "when ",
    "which ",
    "how many ",
    "in what year ",
)

REJECTED_QUESTION_PREFIXES = (
    "why ",
    "how did ",
    "how does ",
    "explain ",
    "compare ",
    "what is the relationship between ",
)

COMPARISON_MARKERS = (
    " older ",
    " younger ",
    " taller ",
    " shorter ",
    " larger ",
    " smaller ",
    " longer ",
    " earlier ",
    " later ",
    " more populous ",
    " less populous ",
    " same nationality",
    " same neighborhood",
    " same country",
    " same state",
    " the same ",
    " both ",
)


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
    titles = query_record["supporting_facts"]["title"]
    return list(dict.fromkeys(titles))


def normalize_text(text: str) -> str:
    """Lowercase and collapse punctuation/whitespace for substring checks."""
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(normalized.split())


def word_count(text: str) -> int:
    """Return the number of whitespace-separated words in `text`."""
    return len([part for part in text.strip().split() if part])


def collect_corpus_titles(corpus_path: Path) -> set[str]:
    """Collect all topic ids available in the source corpus."""
    return {
        record["topic_id"]
        for record in iter_jsonl(corpus_path)
        if record.get("topic_id")
    }


def collect_docs_by_title(corpus_path: Path, titles: set[str]) -> dict[str, dict[str, Any]]:
    """Collect source documents for the requested topic ids."""
    docs_by_title: dict[str, dict[str, Any]] = {}

    for record in iter_jsonl(corpus_path):
        topic_id = record.get("topic_id")
        if topic_id in titles and topic_id not in docs_by_title:
            docs_by_title[topic_id] = record

        if len(docs_by_title) >= len(titles):
            break

    return docs_by_title


def collect_support_docs_and_distractors(
    corpus_path: Path,
    support_titles: set[str],
    distractor_count: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Collect selected support documents and deterministic distractors."""
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


def is_boolean_answer(answer: str) -> bool:
    """Return whether the answer is exactly yes or no."""
    return normalize_text(answer) in {"yes", "no"}


def is_numeric_answer(answer: str) -> bool:
    """Return whether the answer is a short numeric span."""
    return bool(re.fullmatch(r"\s*[0-9][0-9,]*(\.[0-9]+)?\s*", answer))


def is_year_or_date_like(answer: str) -> bool:
    """Return whether the answer looks like a short year or date span."""
    stripped = answer.strip()
    lowered = stripped.lower()

    if re.fullmatch(r"\d{4}", stripped):
        return True

    if any(month in lowered for month in MONTH_NAMES):
        return (
            word_count(stripped) <= MAX_SHORT_ANSWER_WORDS
            and stripped.count(",") <= 1
        )

    return False


def is_time_range_answer(answer: str) -> bool:
    """Return whether the answer looks like a range rather than a span."""
    lowered = answer.lower()
    if re.search(r"\bfrom\s+\d{3,4}\s+to\s+\d{3,4}\b", lowered):
        return True
    if re.search(r"\bbetween\s+.+\s+and\s+.+", lowered):
        return True
    if re.search(r"\d{3,4}\s*[-–]\s*\d{2,4}", answer):
        return True
    return False


def detect_answer_type(question: str, answer: str) -> str:
    """Return a coarse answer-shape label for auditing."""
    if not answer.strip():
        return "empty"
    if is_boolean_answer(answer):
        return "boolean"
    if is_time_range_answer(answer):
        return "time_range"
    if is_numeric_answer(answer):
        return "numeric"
    if is_year_or_date_like(answer):
        return "date_like"

    question_lower = question.strip().lower()
    if question_lower.startswith("who "):
        return "person_or_role"
    if question_lower.startswith("where "):
        return "location"
    if "organization" in question_lower:
        return "organization"
    return "atomic_phrase"


def explain_answer_shape(question: str, answer: str) -> tuple[bool, str | None, str]:
    """Return whether an answer shape is acceptable and, if not, why."""
    stripped = answer.strip()
    detected_answer_type = detect_answer_type(question, stripped)
    lowered = stripped.lower()

    if not stripped:
        return False, "empty_answer", detected_answer_type

    if is_boolean_answer(stripped):
        return True, None, detected_answer_type

    if is_time_range_answer(stripped):
        return False, "time_range_answer", detected_answer_type

    if any(token in stripped for token in (";", "/", "(", ")", "[", "]", "|")):
        return False, "complex_answer_punctuation", detected_answer_type

    if stripped.count(",") > MAX_ALLOWED_COMMAS:
        return False, "complex_answer_punctuation", detected_answer_type

    if re.search(r"\s+or\s+", lowered):
        return False, "ambiguous_answer_shape", detected_answer_type

    if re.search(r"\band\b", lowered) and "," in stripped:
        return False, "list_answer", detected_answer_type

    if stripped.endswith((".", "!", "?")):
        return False, "ambiguous_answer_shape", detected_answer_type

    if word_count(stripped) > MAX_SHORT_ANSWER_WORDS and not (is_numeric_answer(stripped) or is_year_or_date_like(stripped)):
        return False, "answer_too_long", detected_answer_type

    if re.search(r"\b(he|she|they|it)\s+(is|was|were|are)\b", lowered):
        return False, "ambiguous_answer_shape", detected_answer_type

    return True, None, detected_answer_type


def question_is_supported(question: str, answer: str) -> tuple[bool, str | None]:
    """Return whether the question form is conservative enough for this dev set."""
    stripped = question.strip()
    lowered = stripped.lower()

    if any(lowered.startswith(prefix) for prefix in REJECTED_QUESTION_PREFIXES):
        return False, "unsupported_question_type"

    if is_boolean_answer(answer):
        if any(lowered.startswith(prefix) for prefix in BOOLEAN_QUESTION_PREFIXES):
            return True, None
        return False, "unsupported_question_type"

    if any(marker in f" {lowered} " for marker in COMPARISON_MARKERS):
        return False, "unsupported_question_type"

    if any(lowered.startswith(prefix) for prefix in PREFERRED_QUESTION_PREFIXES):
        return True, None

    return False, "unsupported_question_type"


def build_rejection_record(
    query: dict[str, Any],
    detected_answer_type: str,
    rejection_reason: str,
    explanation: str,
) -> dict[str, Any]:
    """Build one rejected-candidate audit row."""
    return {
        "query_id": query["query_id"],
        "question": query["question"],
        "answer": query["answer"],
        "detected_answer_type": detected_answer_type,
        "supporting_facts": query["supporting_facts"],
        "rejection_reason": rejection_reason,
        "explanation": explanation,
    }


def build_accepted_audit_record(
    query: dict[str, Any],
    detected_answer_type: str,
    support_titles: list[str],
    answer_found_in_support: bool,
) -> dict[str, Any]:
    """Build one accepted-candidate audit row."""
    return {
        "query_id": query["query_id"],
        "question": query["question"],
        "answer": query["answer"],
        "detected_answer_type": detected_answer_type,
        "answer_length_words": word_count(query["answer"]),
        "support_titles_found": support_titles,
        "answer_found_in_support": answer_found_in_support,
        "accepted": True,
    }


def classify_candidates(
    *,
    queries: list[dict[str, Any]],
    corpus_titles: set[str],
    docs_by_title: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify source queries into eligible candidates and rejected candidates."""
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_questions: set[str] = set()

    for query in queries:
        normalized_question = normalize_text(query["question"])
        support_titles = unique_support_titles(query)
        answer = query["answer"]

        if normalized_question in seen_questions:
            rejected.append(
                build_rejection_record(
                    query,
                    detect_answer_type(query["question"], answer),
                    "duplicate_question",
                    "Duplicate normalized question text.",
                )
            )
            continue
        seen_questions.add(normalized_question)

        answer_ok, answer_reason, detected_answer_type = explain_answer_shape(
            query["question"],
            answer,
        )
        if not answer_ok:
            rejected.append(
                build_rejection_record(
                    query,
                    detected_answer_type,
                    answer_reason or "ambiguous_answer_shape",
                    "Answer shape is not conservative enough for substring-based evaluation.",
                )
            )
            continue

        question_ok, question_reason = question_is_supported(query["question"], answer)
        if not question_ok:
            rejected.append(
                build_rejection_record(
                    query,
                    detected_answer_type,
                    question_reason or "unsupported_question_type",
                    "Question form is outside the conservative close-ended subset.",
                )
            )
            continue

        if not support_titles or any(title not in corpus_titles for title in support_titles):
            rejected.append(
                build_rejection_record(
                    query,
                    detected_answer_type,
                    "support_not_found_in_corpus",
                    "One or more supporting titles are missing from the local corpus.",
                )
            )
            continue

        answer_found_in_support = False
        if not is_boolean_answer(answer):
            normalized_answer = normalize_text(answer)
            answer_found_in_support = any(
                normalized_answer
                and normalized_answer in normalize_text(docs_by_title[title]["text"])
                for title in support_titles
                if title in docs_by_title
            )
            if not answer_found_in_support:
                rejected.append(
                    build_rejection_record(
                        query,
                        detected_answer_type,
                        "answer_not_found_in_support",
                        "Gold answer does not appear in the available supporting documents.",
                    )
                )
                continue

        eligible.append(
            {
                "query": query,
                "audit": build_accepted_audit_record(
                    query,
                    detected_answer_type,
                    support_titles,
                    answer_found_in_support,
                ),
            }
        )

    return eligible, rejected


def select_close_ended_queries(
    *,
    eligible_candidates: list[dict[str, Any]],
    initial_candidate_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the first deterministic subset of eligible candidates."""
    selected = eligible_candidates[:initial_candidate_count]
    remaining = eligible_candidates[initial_candidate_count:]

    if len(selected) < initial_candidate_count:
        raise ValueError(
            "Only found "
            f"{len(selected)} eligible close-ended queries; requested {initial_candidate_count}."
        )

    return selected, remaining


def build_close_ended_query_set(
    *,
    corpus_path: Path,
    queries_path: Path,
    output_corpus_path: Path,
    output_queries_path: Path,
    accepted_audit_path: Path,
    rejected_audit_path: Path,
    manifest_path: Path,
    initial_candidate_count: int,
    final_target_count: int,
    distractor_count: int,
    seed: int,
) -> None:
    """Build and write the clean close-ended query subset and aligned corpus."""
    print(f"Loading source queries from {queries_path}...")
    queries = list(iter_jsonl(queries_path))
    print(f"Loaded source questions: {len(queries)}")

    print(f"Loading corpus title set from {corpus_path}...")
    corpus_titles = collect_corpus_titles(corpus_path)
    print(f"Found {len(corpus_titles)} unique corpus titles.")

    provisional_support_titles: set[str] = set()
    for query in queries:
        answer_ok, _, _ = explain_answer_shape(query["question"], query["answer"])
        question_ok, _ = question_is_supported(query["question"], query["answer"])
        support_titles = unique_support_titles(query)
        if answer_ok and question_ok and support_titles and all(
            title in corpus_titles for title in support_titles
        ):
            provisional_support_titles.update(support_titles)

    print("Collecting support documents needed for answer-evidence checks...")
    docs_by_title = collect_docs_by_title(corpus_path, provisional_support_titles)

    print("Classifying close-ended candidates...")
    eligible_candidates, rejected_candidates = classify_candidates(
        queries=queries,
        corpus_titles=corpus_titles,
        docs_by_title=docs_by_title,
    )

    selected_candidates, unselected_candidates = select_close_ended_queries(
        eligible_candidates=eligible_candidates,
        initial_candidate_count=initial_candidate_count,
    )
    selected_queries = [candidate["query"] for candidate in selected_candidates]
    accepted_audit_records = [candidate["audit"] for candidate in selected_candidates]

    ordered_support_titles: list[str] = []
    for query in selected_queries:
        for title in unique_support_titles(query):
            if title not in ordered_support_titles:
                ordered_support_titles.append(title)

    support_title_set = set(ordered_support_titles)
    print(
        f"Selected {len(selected_queries)} close-ended queries with "
        f"{len(support_title_set)} unique support docs."
    )

    print(f"Collecting support docs and {distractor_count} random distractors...")
    support_docs, distractors = collect_support_docs_and_distractors(
        corpus_path=corpus_path,
        support_titles=support_title_set,
        distractor_count=distractor_count,
        seed=seed,
    )

    output_corpus = [support_docs[title] for title in ordered_support_titles]
    output_corpus.extend(distractors)

    write_jsonl(output_queries_path, selected_queries)
    write_jsonl(output_corpus_path, output_corpus)
    write_jsonl(accepted_audit_path, accepted_audit_records)
    write_jsonl(rejected_audit_path, rejected_candidates)

    answer_type_distribution = Counter(
        record["detected_answer_type"] for record in accepted_audit_records
    )
    rejection_reason_counts = Counter(
        record["rejection_reason"] for record in rejected_candidates
    )

    manifest = {
        "dataset_name": "close_ended_dev",
        "stage": "initial",
        "script_name": Path(__file__).name,
        "source_dataset": "clean_complete",
        "source_question_count": len(queries),
        "eligible_question_count": len(eligible_candidates),
        "selected_initial_candidate_count": len(selected_queries),
        "rejected_question_count": len(rejected_candidates),
        "not_selected_question_count": len(unselected_candidates),
        "requested_initial_candidate_count": initial_candidate_count,
        "planned_final_target_count": final_target_count,
        "random_seed": seed,
        "distractor_count": len(distractors),
        "support_doc_count": len(ordered_support_titles),
        "corpus_record_count": len(output_corpus),
        "answer_type_distribution": dict(sorted(answer_type_distribution.items())),
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "source_corpus_path": str(corpus_path),
        "source_queries_path": str(queries_path),
        "output_corpus_path": str(output_corpus_path),
        "output_queries_path": str(output_queries_path),
        "accepted_audit_path": str(accepted_audit_path),
        "rejected_audit_path": str(rejected_audit_path),
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

    top_rejections = ", ".join(
        f"{reason}={count}"
        for reason, count in rejection_reason_counts.most_common(5)
    ) or "none"

    print(f"Accepted close-ended questions: {len(selected_queries)}")
    print(f"Rejected questions: {len(rejected_candidates)}")
    print(f"Top rejection reasons: {top_rejections}")
    print(f"Saved close-ended queries: {len(selected_queries)} -> {output_queries_path}")
    print(f"Saved close-ended corpus: {len(output_corpus)} -> {output_corpus_path}")
    print(f"Saved accepted audit -> {accepted_audit_path}")
    print(f"Saved rejected audit -> {rejected_audit_path}")
    print(f"Saved manifest -> {manifest_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Build a clean close-ended HotpotQA development subset."
    )
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--queries-path", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--output-corpus-path", type=Path, default=DEFAULT_OUTPUT_CORPUS_PATH)
    parser.add_argument("--output-queries-path", type=Path, default=DEFAULT_OUTPUT_QUERIES_PATH)
    parser.add_argument("--accepted-audit-path", type=Path, default=DEFAULT_ACCEPTED_AUDIT_PATH)
    parser.add_argument("--rejected-audit-path", type=Path, default=DEFAULT_REJECTED_AUDIT_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--initial-candidate-count",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--final-target-count",
        type=int,
        default=DEFAULT_FINAL_TARGET_COUNT,
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=None,
        help="Deprecated alias for --initial-candidate-count.",
    )
    parser.add_argument("--distractor-count", type=int, default=DEFAULT_DISTRACTOR_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    initial_candidate_count = args.initial_candidate_count
    if args.query_count is not None and initial_candidate_count is not None:
        if args.query_count != initial_candidate_count:
            raise ValueError(
                "--query-count and --initial-candidate-count disagree. "
                "Use only one value."
            )
    if initial_candidate_count is None:
        initial_candidate_count = args.query_count
    if initial_candidate_count is None:
        initial_candidate_count = DEFAULT_INITIAL_CANDIDATE_COUNT
    build_close_ended_query_set(
        corpus_path=args.corpus_path,
        queries_path=args.queries_path,
        output_corpus_path=args.output_corpus_path,
        output_queries_path=args.output_queries_path,
        accepted_audit_path=args.accepted_audit_path,
        rejected_audit_path=args.rejected_audit_path,
        manifest_path=args.manifest_path,
        initial_candidate_count=initial_candidate_count,
        final_target_count=args.final_target_count,
        distractor_count=args.distractor_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
