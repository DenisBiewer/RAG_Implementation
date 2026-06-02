from __future__ import annotations

"""
CLI wrapper around `app.experiment_runner.run_experiments`.

This script exposes dataset presets that match the repository's processed and
built artifact layout, while still delegating the actual batch execution to the
shared experiment runner.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.dataset_registry import (
    CLOSE_ENDED_DATASET_NAMES,
    attack_built_dir,
    attack_logs_dir,
    clean_experiment_index_paths,
    clean_logs_stage_dir,
    clean_processed_stage_dir,
    exfiltration_control_dataset_name,
)
from app.experiment_configs import ALL_CONFIG_NAMES
from app.path_utils import DATA_DIR, LOG_DIR


@dataclass(frozen=True)
class DatasetProfile:
    """Input and output locations for one experiment dataset profile."""

    queries_path: Path
    index_path: Path
    meta_path: Path
    log_dir: Path


DATASET_PROFILES: dict[str, DatasetProfile] = {
    "clean_complete": DatasetProfile(
        queries_path=DATA_DIR / "processed" / "clean" / "complete" / "queries.jsonl",
        index_path=DATA_DIR / "built" / "clean" / "complete" / "faiss.index",
        meta_path=DATA_DIR / "built" / "clean" / "complete" / "faiss_meta.json",
        log_dir=LOG_DIR / "clean" / "complete",
    ),
}

for dataset in CLOSE_ENDED_DATASET_NAMES:
    for stage in ("initial", "final"):
        index_path, meta_path = clean_experiment_index_paths(dataset, stage)
        DATASET_PROFILES[f"{dataset}_{stage}"] = DatasetProfile(
            queries_path=clean_processed_stage_dir(dataset, stage) / "queries.jsonl",
            index_path=index_path,
            meta_path=meta_path,
            log_dir=clean_logs_stage_dir(dataset, stage),
        )

    DATASET_PROFILES[f"instruction_following_{dataset}"] = DatasetProfile(
        queries_path=clean_processed_stage_dir(dataset, "final") / "queries.jsonl",
        index_path=attack_built_dir("instruction_following", dataset) / "faiss.index",
        meta_path=attack_built_dir("instruction_following", dataset) / "faiss_meta.json",
        log_dir=attack_logs_dir("instruction_following", dataset),
    )
    DATASET_PROFILES[f"exfiltration_{dataset}"] = DatasetProfile(
        queries_path=clean_processed_stage_dir(dataset, "final") / "queries.jsonl",
        index_path=attack_built_dir("exfiltration", dataset) / "faiss.index",
        meta_path=attack_built_dir("exfiltration", dataset) / "faiss_meta.json",
        log_dir=attack_logs_dir("exfiltration", dataset),
    )

    control_dataset = exfiltration_control_dataset_name(dataset)
    DATASET_PROFILES[f"exfiltration_{control_dataset}"] = DatasetProfile(
        queries_path=clean_processed_stage_dir(dataset, "final") / "queries.jsonl",
        index_path=attack_built_dir("exfiltration", control_dataset) / "faiss.index",
        meta_path=attack_built_dir("exfiltration", control_dataset) / "faiss_meta.json",
        log_dir=attack_logs_dir("exfiltration", control_dataset),
    )
    DATASET_PROFILES[f"poisoned_rag_{dataset}"] = DatasetProfile(
        queries_path=DATA_DIR / "processed" / "attacks" / "poisoned_rag" / dataset / "final" / "queries.jsonl",
        index_path=attack_built_dir("poisoned_rag", dataset) / "faiss.index",
        meta_path=attack_built_dir("poisoned_rag", dataset) / "faiss_meta.json",
        log_dir=attack_logs_dir("poisoned_rag", dataset),
    )

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


def maybe_clear_outputs(output_paths: dict[str, Path], *, overwrite_existing_logs: bool) -> None:
    """Delete selected output files before a run when explicitly requested."""
    if not overwrite_existing_logs:
        return

    for path in output_paths.values():
        if path.exists():
            path.unlink()
            print(f"Deleted existing log file: {path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the preset-based experiment launcher."""
    parser = argparse.ArgumentParser(
        description="Run experiment batches from named dataset presets."
    )
    parser.add_argument(
        "--dataset-profile",
        choices=sorted(DATASET_PROFILES),
        default="close_ended_main_final",
        help="Named dataset preset to run.",
    )
    parser.add_argument(
        "--config",
        dest="config_names",
        action="append",
        default=None,
        help="Config name to run. Repeat to run multiple configs, or pass --config all.",
    )
    parser.add_argument(
        "--profile",
        default="dev",
        help="Generator profile name from app/model_profiles.py.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of queries to run.",
    )
    parser.add_argument(
        "--overwrite-existing-logs",
        dest="overwrite_existing_logs",
        action="store_true",
        default=True,
        help="Delete selected per-config output files before running.",
    )
    parser.add_argument(
        "--keep-existing-logs",
        dest="overwrite_existing_logs",
        action="store_false",
        help="Keep existing per-config output files instead of deleting them first.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    from app.experiment_runner import run_experiments

    dataset = DATASET_PROFILES[args.dataset_profile]
    config_names = resolve_config_names(args.config_names or ["baseline"])
    output_paths = build_output_paths(config_names, dataset)
    dataset.log_dir.mkdir(parents=True, exist_ok=True)
    maybe_clear_outputs(output_paths, overwrite_existing_logs=args.overwrite_existing_logs)

    print("Experiment launcher configuration:")
    print(f"- dataset: {args.dataset_profile}")
    print(f"- configs: {config_names}")
    print(f"- generator profile: {args.profile}")
    print(f"- limit: {args.limit}")
    print(f"- log directory: {dataset.log_dir}")

    run_experiments(
        queries_path=dataset.queries_path,
        index_path=dataset.index_path,
        meta_path=dataset.meta_path,
        output_path=dataset.log_dir / "combined_runs.jsonl",
        output_paths_by_config=output_paths,
        config_names=config_names,
        profile_name=args.profile,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
