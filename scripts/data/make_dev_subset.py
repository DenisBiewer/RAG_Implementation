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

from app.io_utils import read_jsonl, write_jsonl

""" script used to make dev corpus for faster testing during development"""

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

COMPLETE_DIR = PROCESSED_DIR / "clean" / "complete"
DEV_DIR = PROCESSED_DIR / "clean" / "dev"

CORPUS_PATH = COMPLETE_DIR / "corpus.jsonl"
QUERIES_PATH = COMPLETE_DIR / "queries.jsonl"

DEV_CORPUS_PATH = DEV_DIR / "corpus.jsonl"
DEV_QUERIES_PATH = DEV_DIR / "queries.jsonl"


def main():
    corpus = read_jsonl(str(CORPUS_PATH))
    queries = read_jsonl(str(QUERIES_PATH))

    # Temporary development subset sizes.
    # These are not thesis-final experimental parameters.
    dev_corpus = corpus[:5000]
    dev_queries = queries[:50]

    write_jsonl(str(DEV_CORPUS_PATH), dev_corpus)
    write_jsonl(str(DEV_QUERIES_PATH), dev_queries)

    print(f"Saved dev corpus: {len(dev_corpus)} docs -> {str(DEV_CORPUS_PATH)}")
    print(f"Saved dev queries: {len(dev_queries)} queries -> {str(DEV_QUERIES_PATH)}")


if __name__ == "__main__":
    main()
