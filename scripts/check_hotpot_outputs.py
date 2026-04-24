from app.io_utils import read_jsonl

from pathlib import Path

""" This script loads and prints corpus records for manual verification """

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR.parent / "data" / "processed"
COMPLETE_DIR = PROCESSED_DIR / "clean" / "complete"

CORPUS_PATH = COMPLETE_DIR / "corpus.jsonl"
QUERIES_PATH = COMPLETE_DIR / "queries.jsonl"


def main():
    corpus = read_jsonl(str(CORPUS_PATH))
    queries = read_jsonl(str(QUERIES_PATH))

    print(f"Loaded corpus records: {len(corpus)}")
    print(f"Loaded query records: {len(queries)}")

    print("\nFirst corpus record:")
    print(corpus[0])

    print("\nFirst query record:")
    print(queries[0])


if __name__ == "__main__":
    main()
