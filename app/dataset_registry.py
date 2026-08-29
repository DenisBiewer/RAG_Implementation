from __future__ import annotations

"""Shared dataset registry and path helpers for close-ended workflows."""

from dataclasses import dataclass
from pathlib import Path

from app.path_utils import BUILT_DIR, LOG_DIR, PROCESSED_DIR, RESULTS_DIR


@dataclass(frozen=True)
class CloseEndedDatasetSpec:
    """Configuration for one clean close-ended dataset workflow."""

    name: str
    use_full_corpus: bool
    initial_candidate_count: int
    final_target_count: int
    distractor_count: int
    require_support_titles_in_corpus: bool


CLOSE_ENDED_DATASET_SPECS: dict[str, CloseEndedDatasetSpec] = {
    "close_ended_dev": CloseEndedDatasetSpec(
        name="close_ended_dev",
        use_full_corpus=False,
        initial_candidate_count=50,
        final_target_count=10,
        distractor_count=500,
        require_support_titles_in_corpus=True,
    ),
    "close_ended_main": CloseEndedDatasetSpec(
        name="close_ended_main",
        use_full_corpus=True,
        initial_candidate_count=200,
        final_target_count=20,
        distractor_count=0,
        require_support_titles_in_corpus=True,
    ),
}

DEFAULT_CLOSE_ENDED_DATASET = "close_ended_dev"
DEFAULT_CLOSE_ENDED_MAIN_DATASET = "close_ended_main"
CLOSE_ENDED_DATASET_NAMES = tuple(sorted(CLOSE_ENDED_DATASET_SPECS))
CLEAN_COMPLETE_DATASET = "complete"


def format_temperature_variant(temperature: float | None) -> str | None:
    """Return the canonical directory label for one sampled-generation temperature."""
    if temperature is None:
        return None

    normalized = float(temperature)
    if normalized <= 0.0:
        return None

    return f"temperature_{normalized:g}".replace(".", "_")


def append_variant_dir(base_dir: Path, variant: str | None = None) -> Path:
    """Append an optional experiment-variant subdirectory to one base path."""
    if not variant:
        return base_dir
    return base_dir / variant


def get_close_ended_dataset_spec(dataset: str) -> CloseEndedDatasetSpec:
    """Resolve one registered close-ended dataset."""
    spec = CLOSE_ENDED_DATASET_SPECS.get(dataset)
    if spec is None:
        valid = ", ".join(sorted(CLOSE_ENDED_DATASET_SPECS))
        raise ValueError(f"Unknown close-ended dataset: {dataset!r}. Expected one of: {valid}.")
    return spec


def clean_processed_dir(dataset: str) -> Path:
    """Return the processed clean dataset directory."""
    return PROCESSED_DIR / "clean" / dataset


def clean_processed_stage_dir(dataset: str, stage: str) -> Path:
    """Return the processed clean dataset stage directory."""
    return clean_processed_dir(dataset) / stage


def clean_built_stage_dir(dataset: str, stage: str) -> Path:
    """Return the built clean dataset stage directory."""
    return BUILT_DIR / "clean" / dataset / stage


def clean_logs_stage_dir(dataset: str, stage: str, variant: str | None = None) -> Path:
    """Return the clean experiment log directory for one dataset stage."""
    return append_variant_dir(LOG_DIR / "clean" / dataset / stage, variant)


def validation_results_dir(
    dataset: str,
    *,
    variant: str | None = None,
    base_dir: Path = RESULTS_DIR,
) -> Path:
    """Return the clean-validation results directory for one dataset."""
    return append_variant_dir(base_dir / "validation" / dataset, variant)


def attack_processed_dir(attack_name: str, dataset: str) -> Path:
    """Return the processed attack dataset directory."""
    return PROCESSED_DIR / "attacks" / attack_name / dataset


def attack_processed_stage_dir(attack_name: str, dataset: str, stage: str) -> Path:
    """Return the processed attack dataset stage directory."""
    return attack_processed_dir(attack_name, dataset) / stage


def attack_built_dir(attack_name: str, dataset: str) -> Path:
    """Return the built attack dataset directory."""
    return BUILT_DIR / "attacks" / attack_name / dataset


def attack_built_stage_dir(attack_name: str, dataset: str, stage: str) -> Path:
    """Return the built attack dataset stage directory."""
    return attack_built_dir(attack_name, dataset) / stage


def attack_logs_dir(attack_name: str, dataset: str, variant: str | None = None) -> Path:
    """Return the attack experiment log directory."""
    return append_variant_dir(LOG_DIR / "attacks" / attack_name / dataset, variant)


def attack_results_dir(
    attack_name: str,
    dataset: str,
    *,
    variant: str | None = None,
    base_dir: Path = RESULTS_DIR,
) -> Path:
    """Return the attack-analysis results directory."""
    return append_variant_dir(base_dir / "attacks" / attack_name / dataset, variant)


def exfiltration_control_dataset_name(dataset: str) -> str:
    """Return the exfiltration control dataset name paired with one base dataset."""
    return f"{dataset}_control"


def clean_retrieval_base_info(dataset: str, stage: str) -> dict[str, str | Path]:
    """
    Return the clean retrieval base that should back one dataset/stage workflow.

    Full-corpus datasets reuse the shared `clean_complete` retrieval artifacts.
    Reduced datasets keep dataset-specific stage artifacts.
    """
    spec = get_close_ended_dataset_spec(dataset)
    if spec.use_full_corpus:
        return {
            "preset_name": "clean_complete",
            "input_path": PROCESSED_DIR / "clean" / CLEAN_COMPLETE_DATASET / "corpus.jsonl",
            "chunks_path": BUILT_DIR / "clean" / CLEAN_COMPLETE_DATASET / "chunks.jsonl",
            "index_path": BUILT_DIR / "clean" / CLEAN_COMPLETE_DATASET / "faiss.index",
            "meta_path": BUILT_DIR / "clean" / CLEAN_COMPLETE_DATASET / "faiss_meta.json",
        }

    return {
        "preset_name": f"{dataset}_{stage}",
        "input_path": clean_processed_stage_dir(dataset, stage) / "corpus.jsonl",
        "chunks_path": clean_built_stage_dir(dataset, stage) / "chunks.jsonl",
        "index_path": clean_built_stage_dir(dataset, stage) / "faiss.index",
        "meta_path": clean_built_stage_dir(dataset, stage) / "faiss_meta.json",
    }


def clean_experiment_index_paths(dataset: str, stage: str) -> tuple[Path, Path]:
    """
    Return the index and metadata paths used for clean experiment runs.

    Full-corpus datasets reuse the `clean_complete` retrieval artifacts.
    Reduced clean datasets keep dataset-specific stage artifacts.
    """
    base_info = clean_retrieval_base_info(dataset, stage)
    return base_info["index_path"], base_info["meta_path"]
