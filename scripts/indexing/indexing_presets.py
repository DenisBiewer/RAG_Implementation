from __future__ import annotations

"""Shared dataset path presets for retrieval artifact builders."""

from pathlib import Path

from app.dataset_registry import (
    CLOSE_ENDED_DATASET_NAMES,
    attack_built_dir,
    attack_processed_dir,
    clean_retrieval_base_info,
    clean_built_stage_dir,
    clean_processed_stage_dir,
    exfiltration_control_dataset_name,
    get_close_ended_dataset_spec,
)
from app.path_utils import BUILT_DIR, PROCESSED_DIR

TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"

INDEXING_PRESETS: dict[str, dict[str, Path | str]] = {
    "clean_complete": {
        "preset_type": "clean",
        "input_path": PROCESSED_DIR / "clean" / "complete" / "corpus.jsonl",
        "chunks_path": BUILT_DIR / "clean" / "complete" / "chunks.jsonl",
        "index_path": BUILT_DIR / "clean" / "complete" / "faiss.index",
        "meta_path": BUILT_DIR / "clean" / "complete" / "faiss_meta.json",
    },
}

for dataset in CLOSE_ENDED_DATASET_NAMES:
    dataset_spec = get_close_ended_dataset_spec(dataset)
    if not dataset_spec.use_full_corpus:
        for stage in ("initial", "final"):
            preset_name = f"{dataset}_{stage}"
            INDEXING_PRESETS[preset_name] = {
                "preset_type": "clean",
                "input_path": clean_processed_stage_dir(dataset, stage) / "corpus.jsonl",
                "chunks_path": clean_built_stage_dir(dataset, stage) / "chunks.jsonl",
                "index_path": clean_built_stage_dir(dataset, stage) / "faiss.index",
                "meta_path": clean_built_stage_dir(dataset, stage) / "faiss_meta.json",
            }

    clean_base = clean_retrieval_base_info(dataset, "final")
    instruction_dataset = dataset
    INDEXING_PRESETS[f"instruction_following_{instruction_dataset}"] = {
        "preset_type": "attack",
        "base_preset": clean_base["preset_name"],
        "input_path": attack_processed_dir("instruction_following", instruction_dataset) / "corpus.jsonl",
        "chunks_path": attack_built_dir("instruction_following", instruction_dataset) / "chunks.jsonl",
        "index_path": attack_built_dir("instruction_following", instruction_dataset) / "faiss.index",
        "meta_path": attack_built_dir("instruction_following", instruction_dataset) / "faiss_meta.json",
    }

    exfiltration_dataset = dataset
    INDEXING_PRESETS[f"exfiltration_{exfiltration_dataset}"] = {
        "preset_type": "attack",
        "base_preset": clean_base["preset_name"],
        "input_path": attack_processed_dir("exfiltration", exfiltration_dataset) / "corpus.jsonl",
        "chunks_path": attack_built_dir("exfiltration", exfiltration_dataset) / "chunks.jsonl",
        "index_path": attack_built_dir("exfiltration", exfiltration_dataset) / "faiss.index",
        "meta_path": attack_built_dir("exfiltration", exfiltration_dataset) / "faiss_meta.json",
    }

    control_dataset = exfiltration_control_dataset_name(dataset)
    INDEXING_PRESETS[f"exfiltration_{control_dataset}"] = {
        "preset_type": "attack",
        "base_preset": clean_base["preset_name"],
        "input_path": attack_processed_dir("exfiltration", control_dataset) / "corpus.jsonl",
        "chunks_path": attack_built_dir("exfiltration", control_dataset) / "chunks.jsonl",
        "index_path": attack_built_dir("exfiltration", control_dataset) / "faiss.index",
        "meta_path": attack_built_dir("exfiltration", control_dataset) / "faiss_meta.json",
    }

    INDEXING_PRESETS[f"poisoned_rag_{dataset}"] = {
        "preset_type": "attack",
        "base_preset": clean_base["preset_name"],
        "input_path": attack_processed_dir("poisoned_rag", dataset) / "final" / "corpus.jsonl",
        "chunks_path": attack_built_dir("poisoned_rag", dataset) / "chunks.jsonl",
        "index_path": attack_built_dir("poisoned_rag", dataset) / "faiss.index",
        "meta_path": attack_built_dir("poisoned_rag", dataset) / "faiss_meta.json",
    }


def resolve_chunk_paths(
    preset_name: str,
    *,
    input_path: Path | None = None,
    chunks_path: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve corpus and chunks paths from a preset plus optional overrides."""
    preset = INDEXING_PRESETS[preset_name]
    return input_path or preset["input_path"], chunks_path or preset["chunks_path"]


def resolve_index_paths(
    preset_name: str,
    *,
    chunks_path: Path | None = None,
    index_path: Path | None = None,
    meta_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve chunks, index, and metadata paths from a preset plus overrides."""
    preset = INDEXING_PRESETS[preset_name]
    return (
        chunks_path or preset["chunks_path"],
        index_path or preset["index_path"],
        meta_path or preset["meta_path"],
    )


def is_attack_preset(preset_name: str) -> bool:
    """Return whether one indexing preset should append to a clean retrieval base."""
    return INDEXING_PRESETS[preset_name].get("preset_type") == "attack"


def resolve_attack_base_paths(preset_name: str) -> tuple[str, Path, Path, Path, Path]:
    """Resolve the clean retrieval base for one attack preset."""
    preset = INDEXING_PRESETS[preset_name]
    if preset.get("preset_type") != "attack":
        raise ValueError(f"Preset {preset_name!r} is not an attack preset.")

    base_preset_name = str(preset["base_preset"])
    base_preset = INDEXING_PRESETS[base_preset_name]
    return (
        base_preset_name,
        base_preset["input_path"],
        base_preset["chunks_path"],
        base_preset["index_path"],
        base_preset["meta_path"],
    )


def resolve_retrieval_artifact_paths(
    preset_name: str,
    *,
    input_path: Path | None = None,
    chunks_path: Path | None = None,
    index_path: Path | None = None,
    meta_path: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Resolve the full retrieval-artifact path set from one preset."""
    resolved_input_path, resolved_chunks_path = resolve_chunk_paths(
        preset_name,
        input_path=input_path,
        chunks_path=chunks_path,
    )
    resolved_chunks_path, resolved_index_path, resolved_meta_path = resolve_index_paths(
        preset_name,
        chunks_path=resolved_chunks_path,
        index_path=index_path,
        meta_path=meta_path,
    )
    return (
        resolved_input_path,
        resolved_chunks_path,
        resolved_index_path,
        resolved_meta_path,
    )
