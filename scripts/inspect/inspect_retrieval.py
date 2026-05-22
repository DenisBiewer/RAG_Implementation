from __future__ import annotations

from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = next(
    parent
    for parent in (BASE_DIR, *BASE_DIR.parents)
    if (parent / "app").exists() and (parent / "data").exists()
)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.corpus_loader import load_queries
from app.embedder import ContrieverEmbedder
from app.vector_index import FaissIndex
from app.retrieval import retrieve_top_k

DATA_DIR = PROJECT_DIR / "data"

QUERIES_PATH = DATA_DIR / "processed" / "clean" / "dev" / "queries.jsonl"
INDEX_PATH = DATA_DIR / "built" / "clean" / "dev" / "faiss.index"
META_PATH = DATA_DIR / "built" / "clean" / "dev" / "faiss_meta.json"
DEVICE = "cpu"


def main():
    queries = load_queries(str(QUERIES_PATH))
    print(f"Loaded {len(queries)} queries.")

    # Pick the first query for now
    query = queries[0]
    print("\nQuery ID:", query.query_id)
    print("Question:", query.question)
    print("Expected answer:", query.answer)
    print("Supporting fact titles:", query.supporting_facts.title)

    embedder = ContrieverEmbedder(device=DEVICE)
    index = FaissIndex.load(str(INDEX_PATH), str(META_PATH))

    query_embedding = embedder.encode([query.question])

    results = retrieve_top_k(index, query_embedding, top_k=5)

    print("\nTop-5 retrieved chunks:\n")
    for item in results:
        print("=" * 80)
        print(f"Rank: {item.rank}")
        print(f"Score: {item.score:.4f}")
        print(f"Chunk ID: {item.chunk_id}")
        print(f"Doc ID: {item.doc_id}")
        print(f"Topic ID: {item.topic_id}")
        print(f"Text preview: {item.text[:500]}")
        print()

    print("=" * 80)


if __name__ == "__main__":
    main()
