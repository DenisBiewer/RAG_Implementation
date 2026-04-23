from __future__ import annotations

"""
Batch experiment runner for the current RAG pipeline.

This module is the main entry point for executing many query/configuration
combinations without paying model and index startup cost for every single run.
It loads the retrieval and generation components once, iterates through the
selected evaluation queries, and appends one JSONL record per completed run.

Why this file exists:
- launching one process per query is too expensive for thesis-scale evaluation
- experiment settings should be selectable from the command line
- each run should be logged in a consistent, analysis-friendly structure

Current scope:
- supports the implemented baseline, retrieval-stage, prompt-stage, and output-stage defense pipeline
- accepts the configuration objects defined in `app.config`
- records support-title coverage diagnostics for each query
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from app.config import (
    BASELINE_CONFIG,
    OUTPUT_ONLY_CONFIG,
    PROMPT_ONLY_CONFIG,
    RETRIEVAL_ONLY_CONFIG,
)
from app.corpus_loader import load_queries
from app.embedder import ContrieverEmbedder
from app.generator import QwenGenerator
from app.logging_utils import append_jsonl
from app.model_profiles import GENERATOR_PROFILES
from app.pipeline import run_query
from app.schemas import RunConfig
from app.vector_index import FaissIndex


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
LOG_DIR = PROJECT_DIR / "logs"

DEFAULT_QUERIES_PATH = DATA_DIR / "processed" / "hotpot_queries_dev.jsonl"
DEFAULT_INDEX_PATH = DATA_DIR / "built" / "faiss_dev.index"
DEFAULT_META_PATH = DATA_DIR / "built" / "faiss_dev_meta.json"
DEFAULT_OUTPUT_PATH = LOG_DIR / "experiment_runs.jsonl"

AVAILABLE_CONFIGS: dict[str, RunConfig] = {
    BASELINE_CONFIG.run_name: BASELINE_CONFIG,
    RETRIEVAL_ONLY_CONFIG.run_name: RETRIEVAL_ONLY_CONFIG,
    PROMPT_ONLY_CONFIG.run_name: PROMPT_ONLY_CONFIG,
    OUTPUT_ONLY_CONFIG.run_name: OUTPUT_ONLY_CONFIG,
}


def resolve_run_configs(config_names: list[str]) -> list[RunConfig]:
    """
    Resolve CLI-provided configuration names into `RunConfig` objects.

    Args:
        config_names: Names of experiment configurations requested by the user.

    Returns:
        A list of resolved `RunConfig` objects in the same order as requested.

    Raises:
        ValueError: If any configuration name is unknown.
    """
    resolved: list[RunConfig] = []

    for name in config_names:
        config = AVAILABLE_CONFIGS.get(name)
        if config is None:
            valid = ", ".join(sorted(AVAILABLE_CONFIGS))
            raise ValueError(f"Unknown config '{name}'. Expected one of: {valid}.")
        resolved.append(config)

    return resolved


def ensure_supported_config(config: RunConfig) -> None:
    """
    Reject configurations whose defense logic is not implemented yet.

    Args:
        config: The experiment configuration that is about to be executed.

    Raises:
        NotImplementedError: If the configuration enables any defense toggle.

    Notes:
        Retrieval-, prompt-, and output-stage defenses are currently supported.
    """
    supported_flags = {
        "use_mmr",
        "use_retrieval_filtering",
        "use_prompt_isolation",
        "use_instruction_hierarchy",
        "use_grounding",
        "use_prompt_context_flags",
        "use_sensitive_info_filter",
        "use_policy_check",
    }

    enabled_flags = {
        name
        for name, enabled in config.defense.model_dump().items()
        if enabled
    }
    unsupported_flags = sorted(enabled_flags - supported_flags)

    if unsupported_flags:
        raise NotImplementedError(
            f"Config '{config.run_name}' enables unsupported defenses: {', '.join(unsupported_flags)}."
        )


def build_support_coverage(record: dict, corpus_topic_ids: set[str]) -> dict:
    """
    Summarize whether a query's supporting titles exist in the corpus and were retrieved.

    Args:
        record: One completed run record produced by the pipeline.
        corpus_topic_ids: Set of topic ids available in the loaded FAISS corpus.

    Returns:
        A dictionary containing per-title boolean checks and a compact summary.

    Why this matters:
        When using a dev subset, some supporting articles may simply be absent
        from the corpus. In those cases, low answer accuracy is partly a corpus
        construction issue rather than purely a retrieval or generation failure.
    """
    support_titles = record["supporting_facts"]["title"]
    retrieved_topic_ids = [chunk["topic_id"] for chunk in record["retrieved_chunks"]]

    titles_in_corpus = {title: title in corpus_topic_ids for title in support_titles}
    titles_retrieved = {title: title in retrieved_topic_ids for title in support_titles}

    return {
        "titles_in_corpus": titles_in_corpus,
        "titles_retrieved": titles_retrieved,
        "summary": {
            "total_support_titles": len(support_titles),
            "support_titles_present_in_corpus": sum(titles_in_corpus.values()),
            "support_titles_retrieved": sum(titles_retrieved.values()),
        },
    }


def run_experiments(
    *,
    queries_path: Path,
    index_path: Path,
    meta_path: Path,
    output_path: Path,
    config_names: list[str],
    profile_name: str,
    limit: int | None = None,
) -> None:
    """
    Execute a batch of RAG runs with one-time component initialization.

    The runner performs the following high-level steps:
    1. resolve the requested experiment configs and generator profile
    2. load queries from disk
    3. load the embedder, FAISS index, and generator exactly once
    4. iterate over each config/query pair
    5. run the baseline pipeline and append the resulting record to JSONL

    Args:
        queries_path: Path to the JSONL file containing evaluation queries.
        index_path: Path to the serialized FAISS index.
        meta_path: Path to the JSON metadata aligned with the FAISS rows.
        output_path: Path to the JSONL file that will receive experiment logs.
        config_names: Names of configs to execute.
        profile_name: Name of the generator profile to use.
        limit: Optional maximum number of queries to run from the input file.

    Raises:
        ValueError: If the generator profile is unknown or no queries are
            selected after loading and optional truncation.
        NotImplementedError: If a requested config enables defenses that are not
            implemented in the pipeline yet.

    Notes:
        The runtime device for the embedder is chosen from local CUDA
        availability. The generator device behavior is determined by the chosen
        generator profile and `QwenGenerator`.
    """
    if profile_name not in GENERATOR_PROFILES:
        valid = ", ".join(sorted(GENERATOR_PROFILES))
        raise ValueError(f"Unknown generator profile '{profile_name}'. Expected one of: {valid}.")

    run_configs = resolve_run_configs(config_names)
    for config in run_configs:
        ensure_supported_config(config)

    profile = GENERATOR_PROFILES[profile_name]
    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading queries from {queries_path}...")
    queries = load_queries(str(queries_path))
    if limit is not None:
        queries = queries[:limit]

    if not queries:
        raise ValueError("No queries selected for execution.")

    print(f"Selected {len(queries)} queries.")
    print(f"Runtime device: {runtime_device}")
    print(
        "Generator profile: "
        f"{profile.name} ({profile.model_name}, quantization={profile.quantization})"
    )

    print("Loading embedder...")
    embedder = ContrieverEmbedder(device=runtime_device)

    print("Loading FAISS index...")
    index = FaissIndex.load(str(index_path), str(meta_path))
    corpus_topic_ids = {chunk["topic_id"] for chunk in index.row_to_chunk.values()}

    print("Loading generator...")
    generator = QwenGenerator.from_profile(profile)

    total_runs = len(queries) * len(run_configs)
    print(f"Running {total_runs} query/config combinations...")

    progress = tqdm(total=total_runs, desc="Experiments", unit="run")

    try:
        for config in run_configs:
            for query in queries:
                record = run_query(
                    query_record=query,
                    embedder=embedder,
                    index=index,
                    generator=generator,
                    config=config,
                )

                record["config_name"] = config.run_name
                record["run_config"] = config.model_dump()
                record["generator_profile"] = {
                    "name": profile.name,
                    "model_name": profile.model_name,
                    "device": profile.device,
                    "quantization": profile.quantization,
                    "max_new_tokens": profile.max_new_tokens,
                }
                record["support_coverage"] = build_support_coverage(record, corpus_topic_ids)

                append_jsonl(str(output_path), record)
                progress.update(1)
    finally:
        progress.close()

    print(f"Saved experiment logs to {output_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the experiment runner.

    Returns:
        The parsed CLI namespace.

    CLI behavior:
        - `--config` may be repeated to request multiple experiment configs
        - `--profile` selects the generator preset from `app.model_profiles`
        - `--limit` truncates the loaded query list for debugging or smoke tests
    """
    parser = argparse.ArgumentParser(
        description="Run RAG experiments with one-time model/index initialization."
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help="Path to the JSONL query file.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to the FAISS index file.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=DEFAULT_META_PATH,
        help="Path to the FAISS metadata JSON file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the JSONL output log file.",
    )
    parser.add_argument(
        "--config",
        dest="config_names",
        action="append",
        default=None,
        help="Config name to run. Repeat the flag to run multiple configs.",
    )
    parser.add_argument(
        "--profile",
        default="main",
        choices=sorted(GENERATOR_PROFILES),
        help="Generator profile to use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of queries to run.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point for batch experiment execution.

    If no config is supplied explicitly, the runner defaults to the `baseline`
    configuration so that a plain invocation remains safe and predictable.
    """
    args = parse_args()
    config_names = args.config_names or ["baseline"]
    run_experiments(
        queries_path=args.queries_path,
        index_path=args.index_path,
        meta_path=args.meta_path,
        output_path=args.output_path,
        config_names=config_names,
        profile_name=args.profile,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
