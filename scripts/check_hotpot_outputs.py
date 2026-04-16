from app.io_utils import read_jsonl

""" This script loads and prints corpus records for manual verification """

def main():
    corpus = read_jsonl("../data/processed/hotpot_corpus.jsonl")
    queries = read_jsonl("../data/processed/hotpot_queries.jsonl")

    print(f"Loaded corpus records: {len(corpus)}")
    print(f"Loaded query records: {len(queries)}")

    print("\nFirst corpus record:")
    print(corpus[0])

    print("\nFirst query record:")
    print(queries[0])


if __name__ == "__main__":
    main()