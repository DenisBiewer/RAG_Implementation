from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.io_utils import read_jsonl
from app.path_utils import PROCESSED_DIR

""" This script loads and prints corpus records for manual verification """

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
