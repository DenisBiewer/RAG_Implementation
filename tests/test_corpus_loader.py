import tempfile
import unittest

from pathlib import Path
import json

import self

from app.corpus_loader import load_documents, load_queries
from app.io_utils import write_jsonl


class MyTestCase(unittest.TestCase):

    def test_load_documents(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "docs.jsonl"

            records = [
                {
                    "doc_id": "wiki::Ada Lovelace",
                    "text": "Ada Lovelace was a mathematician.",
                    "document_type": "wiki_article",
                    "topic_id": "Ada Lovelace",
                    "timestamp": None,
                    "is_malicious": False,
                    "attack_type": None,
                }
            ]

            write_jsonl(file_path, records)

            docs = load_documents(str(file_path))

            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].doc_id, "wiki::Ada Lovelace")
            self.assertEqual(docs[0].text, "Ada Lovelace was a mathematician.")
            self.assertEqual(docs[0].document_type, "wiki_article")
            self.assertFalse(docs[0].is_malicious)

    def test_load_queries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "queries.jsonl"

            records = [
                {
                    "query_id": "q1",
                    "question": "Who was Ada Lovelace?",
                    "answer": "A mathematician",
                    "supporting_facts": {
                        "title": ["Ada Lovelace"],
                        "sent_id": [0],
                    },
                }
            ]

            write_jsonl(file_path, records)

            queries = load_queries(str(file_path))

            self.assertEqual(len(queries), 1)
            self.assertEqual(queries[0].query_id, "q1")
            self.assertEqual(queries[0].question, "Who was Ada Lovelace?")
            self.assertEqual(queries[0].supporting_facts.title, ["Ada Lovelace"])
            self.assertEqual(queries[0].supporting_facts.sent_id, [0])

if __name__ == '__main__':
    unittest.main()
