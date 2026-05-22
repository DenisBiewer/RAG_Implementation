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
from app.prompt_builder import build_baseline_messages

DATA_DIR = PROJECT_DIR / "data"

QUERIES_PATH = DATA_DIR / "processed" / "clean" / "dev" / "queries.jsonl"
INDEX_PATH = DATA_DIR / "built" / "clean" / "dev" / "faiss.index"
META_PATH = DATA_DIR / "built" / "clean" / "dev" / "faiss_meta.json"
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
