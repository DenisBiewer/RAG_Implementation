from __future__ import annotations

from pathlib import Path
import sys

import argparse
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.corpus_loader import load_queries
from app.embedder import ContrieverEmbedder
from app.vector_index import FaissIndex
from app.generator import QwenGenerator
from app.model_profiles import GENERATOR_PROFILES
from app.pipeline import run_baseline_query
from app.logging_utils import append_jsonl
from app.seed_config import resolve_seed
from scripts.run.run_experiments import DATASET_PROFILES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the single-query helper."""
    parser = argparse.ArgumentParser(description="Run one baseline query from a dataset preset.")
    parser.add_argument(
        "--dataset-profile",
        choices=sorted(DATASET_PROFILES),
        default="close_ended_main_final",
        help="Named dataset preset to sample the first query from.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(GENERATOR_PROFILES),
        default="dev",
        help="Generator profile name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = DATASET_PROFILES[args.dataset_profile]
    profile = GENERATOR_PROFILES[args.profile]
    effective_seed = resolve_seed()
    queries = load_queries(str(dataset.queries_path))
    query = queries[0]

    print("Loading embedder...")
    print(f"Runtime device: {DEVICE}")
    embedder = ContrieverEmbedder(device=DEVICE)

    print("Loading FAISS index...")
    index = FaissIndex.load(str(dataset.index_path), str(dataset.meta_path))

    print("Loading generator...")
    print(
        f"Generator profile: {profile.name} "
        f"({profile.model_name}, quantization={profile.quantization})"
    )
    generator = QwenGenerator.from_profile(profile)

    print("\nRunning baseline query...")
    record = run_baseline_query(
        query_record=query,
        embedder=embedder,
        index=index,
        generator=generator,
        top_k=5,
        seed=effective_seed,
    )

    log_path = dataset.log_dir / "baseline_runs.jsonl"
    append_jsonl(str(log_path), record)

    print("\nQUESTION:")
    print(record["query"])

    print("\nEXPECTED ANSWER:")
    print(record["expected_answer"])

    print("\nRETRIEVED TOPICS:")
    for chunk in record["retrieved_chunks"]:
        print("-", chunk["topic_id"])

    print("\nMODEL OUTPUT:")
    print(record["model_output"])

    print("\nLATENCIES (ms):")
    print(record["latencies_ms"])

    print(f"\nSaved log record to {str(log_path)}")


if __name__ == "__main__":
    main()
