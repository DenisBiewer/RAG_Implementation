from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys

from datasets import load_dataset


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.io_utils import write_jsonl
from app.path_utils import PROCESSED_DIR
from app.schemas import Document, QueryRecord

""" 
This script is responsible for loading the HotpotQA and fulltext datasets and 
storing them in JSONl in the expected data structure.
"""

COMPLETE_OUTPUT_DIR = PROCESSED_DIR / "clean" / "complete"
CORPUS_OUTPUT_PATH = COMPLETE_OUTPUT_DIR / "corpus.jsonl"
QUERIES_OUTPUT_PATH = COMPLETE_OUTPUT_DIR / "queries.jsonl"


def build_corpus(dataset) -> list[Document]:
    """
    Extract Wikipedia articles from HotpotQA context fields and deduplicate by title.

    Each context entry looks like:
        [title, [sentence1, sentence2, ...]]

    We convert this into a plain-text document aligned with the thesis system model.
    """
    unique_docs: OrderedDict[str, Document] = OrderedDict()

    for split_name in ["train", "validation"]:
        for item in dataset[split_name]:
            context = item["context"]
            titles = context["title"]
            sentences_list = context["sentences"]
            for title, sentences in zip(titles, sentences_list):
                # format text into one clean, coherent string
                text = " ".join(s.strip() for s in sentences if s and s.strip())
                if not text:
                    continue

                doc_id = f"wiki::{title}"

                # Deduplicate by doc_id/title
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = Document(
                        doc_id=doc_id,
                        text=text,
                        document_type="wiki_article",
                        topic_id=title,
                        timestamp=None,
                        is_malicious=False,
                        attack_type=None,
                    )

    return sorted(unique_docs.values(), key=lambda doc: doc.doc_id)


def build_hotpot_queries(dataset, split: str = "validation") -> list[QueryRecord]:
    """
    Build hotpot queries from HotpotQA question-answer pairs.

    These queries are used for:
        - baseline correctness
        - defense side effects (unexpected negative effects)
        - false positives
        - latency

    We use the validation split as the default evaluation pool.
    """
    queries: list[QueryRecord] = []

    for item in dataset[split]:
        queries.append(
            QueryRecord(
                query_id=item["id"],
                question=item["question"],
                answer=item["answer"],
                supporting_facts=item["supporting_facts"],
            )
        )

    return sorted(queries, key=lambda query: query.query_id)


def main():
    print("Loading HotpotQA, fullwiki dataset...")
    dataset = load_dataset("hotpot_qa", "fullwiki")

    print("Building corpus...")
    corpus = build_corpus(dataset)

    print("Building hotpot queries...")
    queries = build_hotpot_queries(dataset, split="validation")

    print(f"Corpus size: {len(corpus)}")
    print(f"Query count: {len(queries)}")

    write_jsonl(str(CORPUS_OUTPUT_PATH), (doc.model_dump() for doc in corpus))
    write_jsonl(str(QUERIES_OUTPUT_PATH), (q.model_dump() for q in queries))

    print(f"Saved corpus to: {str(CORPUS_OUTPUT_PATH)}")
    print(f"Saved queries to: {str(QUERIES_OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
