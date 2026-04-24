from __future__ import annotations

"""
PyCharm-friendly experiment launcher.

Edit only the small configuration block below for normal use:
- `DATASET_PROFILE`: choose a clean or attacked dataset profile
- `CONFIG_NAMES`: choose one or more experiment configs, or use `["all"]`
- `PROFILE_NAME`: choose the generator profile
- `LIMIT`: optionally cap the number of queries for quick checks

Each selected config writes to its own JSONL file under:
    logs/clean/<dataset>/
or:
    logs/attacks/<attack>/<dataset>/

The underlying runner still loads the embedder, index, and generator only once.
"""

from dataclasses import dataclass
from pathlib import Path

from app.experiment_runner import run_experiments


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
LOG_DIR = PROJECT_DIR / "logs"


@dataclass(frozen=True)
class DatasetProfile:
    """Input and output locations for one experiment dataset profile."""

    queries_path: Path
    index_path: Path
    meta_path: Path
    log_dir: Path


DATASET_PROFILES: dict[str, DatasetProfile] = {
    "clean_dev": DatasetProfile(
        queries_path=DATA_DIR / "processed" / "clean" / "dev" / "queries.jsonl",
        index_path=DATA_DIR / "built" / "clean" / "dev" / "faiss.index",
        meta_path=DATA_DIR / "built" / "clean" / "dev" / "faiss_meta.json",
        log_dir=LOG_DIR / "clean" / "dev",
    ),
    "clean_controlled_dev": DatasetProfile(
        queries_path=DATA_DIR / "processed" / "clean" / "controlled_dev" / "queries.jsonl",
        index_path=DATA_DIR / "built" / "clean" / "controlled_dev" / "faiss.index",
        meta_path=DATA_DIR / "built" / "clean" / "controlled_dev" / "faiss_meta.json",
        log_dir=LOG_DIR / "clean" / "controlled_dev",
    ),
    "clean_complete": DatasetProfile(
        queries_path=DATA_DIR / "processed" / "clean" / "complete" / "queries.jsonl",
        index_path=DATA_DIR / "built" / "clean" / "complete" / "faiss.index",
        meta_path=DATA_DIR / "built" / "clean" / "complete" / "faiss_meta.json",
        log_dir=LOG_DIR / "clean" / "complete",
    ),
    "instruction_following_controlled_dev": DatasetProfile(
        queries_path=DATA_DIR / "processed" / "clean" / "controlled_dev" / "queries.jsonl",
        index_path=DATA_DIR / "built" / "attacks" / "instruction_following" / "controlled_dev" / "faiss.index",
        meta_path=DATA_DIR / "built" / "attacks" / "instruction_following" / "controlled_dev" / "faiss_meta.json",
        log_dir=LOG_DIR / "attacks" / "instruction_following" / "controlled_dev",
    ),
}

ALL_CONFIG_NAMES = [
    "baseline",
    "retrieval_only",
    "prompt_only",
    "output_only",
]


# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------
# Dataset choices:
# - "clean_dev"
# - "clean_controlled_dev"
# - "clean_complete"
# - "instruction_following_controlled_dev"
DATASET_PROFILE = "instruction_following_controlled_dev"

# Config choices: any subset of ALL_CONFIG_NAMES, or ["all"]
CONFIG_NAMES = ["all"]

# Generator choices come from app/model_profiles.py: "dev", "main", "validation"
PROFILE_NAME = "dev"

# Use an integer for smoke tests, or None to run the full selected dataset.
LIMIT: int | None = None

# If True, existing output files for the selected dataset/configs are deleted first.
OVERWRITE_EXISTING_LOGS = True


def resolve_config_names(config_names: list[str]) -> list[str]:
    """Normalize selected config names and validate them before running."""
    if config_names == ["all"]:
        return list(ALL_CONFIG_NAMES)

    unknown = sorted(set(config_names) - set(ALL_CONFIG_NAMES))
    if unknown:
        valid = ", ".join(ALL_CONFIG_NAMES + ["all"])
        raise ValueError(f"Unknown config(s): {', '.join(unknown)}. Expected one of: {valid}.")

    return config_names


def build_output_paths(config_names: list[str], dataset: DatasetProfile) -> dict[str, Path]:
    """Build one output JSONL path per experiment config."""
    return {
        config_name: dataset.log_dir / f"{config_name}_runs.jsonl"
        for config_name in config_names
    }


def maybe_clear_outputs(output_paths: dict[str, Path]) -> None:
    """Delete selected output files before a run when explicitly requested."""
    if not OVERWRITE_EXISTING_LOGS:
        return

    for path in output_paths.values():
        if path.exists():
            path.unlink()
            print(f"Deleted existing log file: {path}")


def main() -> None:
    """Run the configured experiment batch."""
    if DATASET_PROFILE not in DATASET_PROFILES:
        valid = ", ".join(sorted(DATASET_PROFILES))
        raise ValueError(f"Unknown dataset profile '{DATASET_PROFILE}'. Expected one of: {valid}.")

    dataset = DATASET_PROFILES[DATASET_PROFILE]
    config_names = resolve_config_names(CONFIG_NAMES)
    output_paths = build_output_paths(config_names, dataset)
    dataset.log_dir.mkdir(parents=True, exist_ok=True)
    maybe_clear_outputs(output_paths)

    print("Experiment launcher configuration:")
    print(f"- dataset: {DATASET_PROFILE}")
    print(f"- configs: {config_names}")
    print(f"- generator profile: {PROFILE_NAME}")
    print(f"- limit: {LIMIT}")
    print(f"- log directory: {dataset.log_dir}")

    run_experiments(
        queries_path=dataset.queries_path,
        index_path=dataset.index_path,
        meta_path=dataset.meta_path,
        output_path=dataset.log_dir / "combined_runs.jsonl",
        output_paths_by_config=output_paths,
        config_names=config_names,
        profile_name=PROFILE_NAME,
        limit=LIMIT,
    )


if __name__ == "__main__":
    main()
