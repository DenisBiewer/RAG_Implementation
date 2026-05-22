from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = next(
    parent
    for parent in (BASE_DIR, *BASE_DIR.parents)
    if (parent / "app").exists() and (parent / "data").exists()
)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.model_profiles import GENERATOR_PROFILES, GeneratorProfile


PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
CLEAN_CLOSE_ENDED_DEV_FINAL_DIR = PROCESSED_DIR / "clean" / "close_ended_dev" / "final"
ATTACK_OUTPUT_DIR = PROCESSED_DIR / "attacks" / "poisoned_rag" / "close_ended_dev"
TARGET_ANSWER_DIR = ATTACK_OUTPUT_DIR / "target_answers"
TARGET_ANSWER_INITIAL_DIR = TARGET_ANSWER_DIR / "initial"
TARGET_ANSWER_FINAL_DIR = TARGET_ANSWER_DIR / "final"
GENERATED_PASSAGES_DIR = ATTACK_OUTPUT_DIR / "generated_passages"
ATTACK_FINAL_DIR = ATTACK_OUTPUT_DIR / "final"

DEFAULT_FINAL_QUERIES_PATH = CLEAN_CLOSE_ENDED_DEV_FINAL_DIR / "queries.jsonl"
DEFAULT_FINAL_CORPUS_PATH = CLEAN_CLOSE_ENDED_DEV_FINAL_DIR / "corpus.jsonl"

DEFAULT_TARGET_ANSWER_REVIEW_PATH = TARGET_ANSWER_INITIAL_DIR / "target_answer_review.jsonl"
DEFAULT_TARGET_ANSWER_MANIFEST_PATH = TARGET_ANSWER_INITIAL_DIR / "manifest.json"
DEFAULT_FINAL_TARGET_ANSWER_QUERIES_PATH = TARGET_ANSWER_FINAL_DIR / "queries.jsonl"
DEFAULT_FINAL_TARGET_ANSWER_MANIFEST_PATH = TARGET_ANSWER_FINAL_DIR / "manifest.json"

DEFAULT_GENERATED_PASSAGES_PATH = GENERATED_PASSAGES_DIR / "passages.jsonl"
DEFAULT_GENERATED_PASSAGES_MANIFEST_PATH = GENERATED_PASSAGES_DIR / "manifest.json"

DEFAULT_ATTACK_QUERIES_PATH = ATTACK_FINAL_DIR / "queries.jsonl"
DEFAULT_ATTACK_CORPUS_PATH = ATTACK_FINAL_DIR / "corpus.jsonl"
DEFAULT_ATTACK_POISON_DOCS_PATH = ATTACK_FINAL_DIR / "poison_docs.jsonl"
DEFAULT_ATTACK_SCENARIOS_PATH = ATTACK_FINAL_DIR / "scenarios.jsonl"
DEFAULT_ATTACK_MANIFEST_PATH = ATTACK_FINAL_DIR / "manifest.json"

ATTACK_TYPE = "poisoned_rag"


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


def resolve_generator_profile(profile_name: str) -> GeneratorProfile:
    """Resolve one configured generator profile by name."""
    profile = GENERATOR_PROFILES.get(profile_name)
    if profile is None:
        valid = ", ".join(sorted(GENERATOR_PROFILES))
        raise ValueError(f"Unknown generator profile: {profile_name!r}. Expected one of: {valid}.")
    return profile
