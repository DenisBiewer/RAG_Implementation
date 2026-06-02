from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.dataset_registry import (
    DEFAULT_CLOSE_ENDED_MAIN_DATASET,
    attack_processed_dir,
    attack_processed_stage_dir,
    clean_processed_stage_dir,
)
from app.io_utils import iter_jsonl, write_jsonl
from app.model_profiles import GENERATOR_PROFILES, GeneratorProfile

ATTACK_TYPE = "poisoned_rag"
DEFAULT_DATASET = DEFAULT_CLOSE_ENDED_MAIN_DATASET


def resolve_poisoned_rag_paths(dataset: str) -> dict[str, Path]:
    """Resolve the processed clean and attack paths for one PoisonedRAG dataset."""
    clean_final_dir = clean_processed_stage_dir(dataset, "final")
    attack_output_dir = attack_processed_dir("poisoned_rag", dataset)
    target_answer_dir = attack_output_dir / "target_answers"
    target_answer_initial_dir = target_answer_dir / "initial"
    target_answer_final_dir = target_answer_dir / "final"
    generated_passages_dir = attack_output_dir / "generated_passages"
    attack_final_dir = attack_processed_stage_dir("poisoned_rag", dataset, "final")

    return {
        "final_queries_path": clean_final_dir / "queries.jsonl",
        "final_corpus_path": clean_final_dir / "corpus.jsonl",
        "target_answer_review_path": target_answer_initial_dir / "target_answer_review.jsonl",
        "target_answer_manifest_path": target_answer_initial_dir / "manifest.json",
        "final_target_answer_queries_path": target_answer_final_dir / "queries.jsonl",
        "final_target_answer_manifest_path": target_answer_final_dir / "manifest.json",
        "generated_passages_path": generated_passages_dir / "passages.jsonl",
        "generated_passages_manifest_path": generated_passages_dir / "manifest.json",
        "attack_queries_path": attack_final_dir / "queries.jsonl",
        "attack_corpus_path": attack_final_dir / "corpus.jsonl",
        "attack_poison_docs_path": attack_final_dir / "poison_docs.jsonl",
        "attack_scenarios_path": attack_final_dir / "scenarios.jsonl",
        "attack_manifest_path": attack_final_dir / "manifest.json",
    }


DEFAULT_PATHS = resolve_poisoned_rag_paths(DEFAULT_DATASET)
DEFAULT_FINAL_QUERIES_PATH = DEFAULT_PATHS["final_queries_path"]
DEFAULT_FINAL_CORPUS_PATH = DEFAULT_PATHS["final_corpus_path"]
DEFAULT_TARGET_ANSWER_REVIEW_PATH = DEFAULT_PATHS["target_answer_review_path"]
DEFAULT_TARGET_ANSWER_MANIFEST_PATH = DEFAULT_PATHS["target_answer_manifest_path"]
DEFAULT_FINAL_TARGET_ANSWER_QUERIES_PATH = DEFAULT_PATHS["final_target_answer_queries_path"]
DEFAULT_FINAL_TARGET_ANSWER_MANIFEST_PATH = DEFAULT_PATHS["final_target_answer_manifest_path"]
DEFAULT_GENERATED_PASSAGES_PATH = DEFAULT_PATHS["generated_passages_path"]
DEFAULT_GENERATED_PASSAGES_MANIFEST_PATH = DEFAULT_PATHS["generated_passages_manifest_path"]
DEFAULT_ATTACK_QUERIES_PATH = DEFAULT_PATHS["attack_queries_path"]
DEFAULT_ATTACK_CORPUS_PATH = DEFAULT_PATHS["attack_corpus_path"]
DEFAULT_ATTACK_POISON_DOCS_PATH = DEFAULT_PATHS["attack_poison_docs_path"]
DEFAULT_ATTACK_SCENARIOS_PATH = DEFAULT_PATHS["attack_scenarios_path"]
DEFAULT_ATTACK_MANIFEST_PATH = DEFAULT_PATHS["attack_manifest_path"]


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
