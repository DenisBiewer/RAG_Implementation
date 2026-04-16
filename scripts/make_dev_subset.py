from app.io_utils import read_jsonl, write_jsonl
""" script used to make dev corpus for faster testing during development"""

DEV_CORPUS_PATH = "../data/processed/hotpot_corpus_dev.jsonl"
DEV_QUERIES_PATH = "../data/processed/hotpot_queries_dev.jsonl"


def main():
    corpus = read_jsonl("../data/processed/hotpot_corpus.jsonl")
    queries = read_jsonl("../data/processed/hotpot_queries.jsonl")

    # Temporary development subset sizes.
    # These are not thesis-final experimental parameters.
    dev_corpus = corpus[:5000]
    dev_queries = queries[:50]

    write_jsonl(DEV_CORPUS_PATH, dev_corpus)
    write_jsonl(DEV_QUERIES_PATH, dev_queries)

    print(f"Saved dev corpus: {len(dev_corpus)} docs -> {DEV_CORPUS_PATH}")
    print(f"Saved dev queries: {len(dev_queries)} queries -> {DEV_QUERIES_PATH}")


if __name__ == "__main__":
    main()