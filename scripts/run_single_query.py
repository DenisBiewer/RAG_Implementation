from __future__ import annotations

from pathlib import Path

import torch

from app.corpus_loader import load_queries
from app.embedder import ContrieverEmbedder
from app.vector_index import FaissIndex
from app.generator import QwenGenerator
from app.model_profiles import DEV_GENERATOR_PROFILE
from app.pipeline import run_baseline_query
from app.logging_utils import append_jsonl


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

QUERIES_PATH = DATA_DIR / "processed/hotpot_queries_dev.jsonl"
INDEX_PATH = DATA_DIR / "built/faiss_dev.index"
META_PATH = DATA_DIR / "built/faiss_dev_meta.json"
LOG_PATH = BASE_DIR.parent / "logs/baseline_runs.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    queries = load_queries(str(QUERIES_PATH))
    query = queries[0]

    print("Loading embedder...")
    print(f"Runtime device: {DEVICE}")
    embedder = ContrieverEmbedder(device=DEVICE)

    print("Loading FAISS index...")
    index = FaissIndex.load(str(INDEX_PATH), str(META_PATH))

    print("Loading generator...")
    print(
        f"Generator profile: {DEV_GENERATOR_PROFILE.name} "
        f"({DEV_GENERATOR_PROFILE.model_name}, quantization={DEV_GENERATOR_PROFILE.quantization})"
    )
    generator = QwenGenerator.from_profile(DEV_GENERATOR_PROFILE)

    print("\nRunning baseline query...")
    record = run_baseline_query(
        query_record=query,
        embedder=embedder,
        index=index,
        generator=generator,
        top_k=5,
        seed=42,
    )

    append_jsonl(str(LOG_PATH), record)

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

    print(f"\nSaved log record to {str(LOG_PATH)}")


if __name__ == "__main__":
    main()
