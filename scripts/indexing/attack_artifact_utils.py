from __future__ import annotations

"""Helpers for append-only attack retrieval-artifact builds."""

import json
from pathlib import Path
from typing import Any, Iterable

from app.io_utils import ensure_parent_dir, iter_jsonl, read_json, write_json


RETRIEVAL_ARTIFACT_COMPATIBILITY_WARNING = (
    "Warning: `--tokenizer-name`, `--chunk-size`, `--overlap`, and the effective "
    "input corpus/order are retrieval-artifact compatibility settings. If you "
    "changed any of these when building the clean dataset, you must use the same "
    "values when building attack artifacts. Otherwise, the attacked artifacts may "
    "no longer be an append-only derivative of the clean retrieval base."
)

PERFORMANCE_SETTINGS_NOTE = (
    "Note: `--device` and `--batch-size` are performance settings rather than "
    "structural compatibility settings. They should not change the logical corpus "
    "structure, but CPU and CUDA runs may not guarantee bit-for-bit identical "
    "embeddings."
)


def build_info_path_for_meta(meta_path: Path) -> Path:
    """Return the sidecar path used to persist retrieval build settings."""
    return meta_path.with_name("retrieval_build_info.json")


def load_build_info(meta_path: Path) -> dict[str, Any] | None:
    """Load retrieval build settings if a sidecar file exists."""
    build_info_path = build_info_path_for_meta(meta_path)
    if not build_info_path.exists():
        return None
    return read_json(build_info_path)


def write_build_info(meta_path: Path, payload: dict[str, Any]) -> None:
    """Persist retrieval build settings next to the FAISS metadata file."""
    write_json(build_info_path_for_meta(meta_path), payload)


def validate_base_build_settings(
    base_build_info: dict[str, Any] | None,
    *,
    tokenizer_name: str,
    chunk_size: int,
    overlap: int,
) -> str | None:
    """
    Validate attack-build compatibility settings against a clean retrieval base.

    Returns a warning string when the base build does not expose persisted build
    settings, so the caller can surface a clear manual-check message.
    """
    if base_build_info is None:
        return (
            "Clean retrieval base build settings were not found on disk. "
            "Automatic compatibility verification is limited to clean-prefix "
            "validation for the effective input corpus/order."
        )

    mismatches: list[str] = []
    expected = {
        "tokenizer_name": tokenizer_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
    labels = {
        "tokenizer_name": "--tokenizer-name",
        "chunk_size": "--chunk-size",
        "overlap": "--overlap",
    }

    for key, expected_value in expected.items():
        base_value = base_build_info.get(key)
        if base_value != expected_value:
            mismatches.append(
                f"{labels[key]} (clean base: {base_value!r}, attack build: {expected_value!r})"
            )

    if mismatches:
        raise ValueError(
            "Retrieval-artifact compatibility settings do not match the clean "
            "retrieval base: "
            + ", ".join(mismatches)
        )

    return None


def ensure_attack_output_paths_distinct(
    *,
    base_chunks_path: Path,
    base_index_path: Path,
    base_meta_path: Path,
    chunks_path: Path,
    index_path: Path,
    meta_path: Path,
) -> None:
    """Reject attack builds that would overwrite the selected clean retrieval base."""
    comparisons = (
        ("chunks", chunks_path, base_chunks_path),
        ("index", index_path, base_index_path),
        ("metadata", meta_path, base_meta_path),
    )

    for label, output_path, base_path in comparisons:
        if output_path.resolve() == base_path.resolve():
            raise ValueError(
                f"Attack {label} output path {output_path} would overwrite the "
                f"clean retrieval base at {base_path}."
            )


def extract_appended_records(
    *,
    base_path: Path,
    attacked_path: Path,
    require_suffix: bool = True,
) -> tuple[int, list[dict[str, Any]]]:
    """
    Validate that one JSONL file is a strict prefix extension of another.

    Returns the clean prefix length plus the appended suffix records.
    """
    base_iter = iter_jsonl(base_path)
    attacked_iter = iter_jsonl(attacked_path)
    clean_prefix_count = 0

    for clean_prefix_count, base_record in enumerate(base_iter, start=1):
        attacked_record = next(attacked_iter, None)
        if attacked_record is None:
            raise ValueError(
                "Attacked corpus is shorter than the clean retrieval base and "
                "cannot be an append-only derivative."
            )
        if attacked_record != base_record:
            raise ValueError(
                "Attacked corpus diverges from the clean retrieval base before "
                f"the appended suffix at record {clean_prefix_count}."
            )

    appended_records = list(attacked_iter)
    if require_suffix and not appended_records:
        raise ValueError(
            "Attacked corpus does not append any records after the clean "
            "retrieval base."
        )

    return clean_prefix_count, appended_records


def write_appended_jsonl(
    *,
    prefix_path: Path,
    output_path: Path,
    suffix_records: Iterable[dict[str, Any]],
) -> None:
    """Write a JSONL file as an existing prefix file plus appended records."""
    ensure_parent_dir(output_path)
    with prefix_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            destination.write(line)
        for record in suffix_records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
