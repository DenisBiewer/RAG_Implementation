from app.io_utils import read_jsonl, write_jsonl

from pathlib import Path

""" script used to make dev corpus for faster testing during development"""

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR.parent / "data" / "processed"

CORPUS_PATH = PROCESSED_DIR / "hotpot_corpus.jsonl"
QUERIES_PATH = PROCESSED_DIR / "hotpot_queries.jsonl"

DEV_CORPUS_PATH = PROCESSED_DIR / "hotpot_corpus_dev.jsonl"
DEV_QUERIES_PATH = PROCESSED_DIR / "hotpot_queries_dev.jsonl"


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