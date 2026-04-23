from __future__ import annotations

from pathlib import Path

from app.corpus_loader import load_queries
from app.embedder import ContrieverEmbedder
from app.vector_index import FaissIndex
from app.retrieval import retrieve_top_k
from app.prompt_builder import build_baseline_messages


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

QUERIES_PATH = DATA_DIR / "processed/hotpot_queries_dev.jsonl"
INDEX_PATH = DATA_DIR / "built/faiss_dev.index"
META_PATH = DATA_DIR / "built/faiss_dev_meta.json"
DEVICE = "cpu"


def main():
    queries = load_queries(str(QUERIES_PATH))
    query = queries[0]

    embedder = ContrieverEmbedder(device=DEVICE)
    index = FaissIndex.load(str(INDEX_PATH), str(META_PATH))

    query_embedding = embedder.encode([query.question])
    retrieved = retrieve_top_k(index, query_embedding, top_k=5)

    prompt = build_baseline_messages(query.question, retrieved)

    print("QUESTION:")
    print(query.question)
    print("\nEXPECTED ANSWER:")
    print(query.answer)
    print("\nPROMPT:\n")
    print(prompt)


if __name__ == "__main__":
    main()