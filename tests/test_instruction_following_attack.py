from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path

from scripts.attacks.build_instruction_following_attack_set import build_attack_set


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class TestInstructionFollowingAttackSet(unittest.TestCase):
    def test_build_attack_set_appends_malicious_docs_after_clean_prefix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_corpus_path = root / "clean" / "corpus.jsonl"
            queries_path = root / "clean" / "queries.jsonl"
            output_corpus_path = root / "attack" / "corpus.jsonl"
            scenarios_path = root / "attack" / "scenarios.jsonl"
            manifest_path = root / "attack" / "manifest.json"

            clean_records = [
                {
                    "doc_id": "wiki::Scott Derrickson",
                    "text": "Scott Derrickson is an American director.",
                    "document_type": "wiki_article",
                    "topic_id": "Scott Derrickson",
                    "timestamp": None,
                    "is_malicious": False,
                    "attack_type": None,
                },
                {
                    "doc_id": "wiki::Ed Wood",
                    "text": "Ed Wood was an American filmmaker.",
                    "document_type": "wiki_article",
                    "topic_id": "Ed Wood",
                    "timestamp": None,
                    "is_malicious": False,
                    "attack_type": None,
                },
            ]
            query_records = [
                {
                    "query_id": "q1",
                    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
                    "answer": "yes",
                    "supporting_facts": {
                        "title": ["Scott Derrickson", "Ed Wood"],
                        "sent_id": [0, 0],
                    },
                }
            ]

            write_jsonl(base_corpus_path, clean_records)
            write_jsonl(queries_path, query_records)

            build_attack_set(
                dataset="close_ended_main",
                base_corpus_path=base_corpus_path,
                queries_path=queries_path,
                output_corpus_path=output_corpus_path,
                scenarios_path=scenarios_path,
                manifest_path=manifest_path,
                sentinel="INJECTION_SUCCESS_001",
            )

            attacked_records = [
                json.loads(line)
                for line in output_corpus_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(attacked_records[: len(clean_records)], clean_records)
            self.assertEqual(len(attacked_records), len(clean_records) + 1)
            self.assertTrue(attacked_records[-1]["is_malicious"])
            self.assertEqual(
                attacked_records[-1]["attack_type"],
                "indirect_instruction_following",
            )
            self.assertTrue(attacked_records[-1]["doc_id"].startswith("attack::"))


if __name__ == "__main__":
    unittest.main()
