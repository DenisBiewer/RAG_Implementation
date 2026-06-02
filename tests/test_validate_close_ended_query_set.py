from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path

from scripts.data.validate_close_ended_query_set import (
    validate_close_ended_query_set,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class TestPrepareCloseEndedQuerySetForFinalization(unittest.TestCase):
    def test_validate_close_ended_query_set_writes_dataset_prep_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queries_path = root / "queries.jsonl"
            logs_dir = root / "logs"
            validated_queries_path = root / "processed" / "validated_queries.jsonl"
            failed_queries_path = root / "processed" / "failed_queries.jsonl"
            export_manifest_path = root / "processed" / "validation_manifest.json"
            manual_review_path = root / "processed" / "manual_review.jsonl"

            source_queries = [
                {
                    "query_id": "q_bool",
                    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
                    "answer": "yes",
                    "supporting_facts": {
                        "title": ["Scott Derrickson", "Ed Wood"],
                        "sent_id": [0, 0],
                    },
                },
                {
                    "query_id": "q_role",
                    "question": "What government position was held by the woman who portrayed Corliss Archer in Kiss and Tell?",
                    "answer": "Chief of Protocol",
                    "supporting_facts": {
                        "title": ["Kiss and Tell (1945 film)", "Shirley Temple"],
                        "sent_id": [0, 0],
                    },
                },
                {
                    "query_id": "q_year",
                    "question": "In what year was Sample Person born?",
                    "answer": "1984",
                    "supporting_facts": {
                        "title": ["Sample Person"],
                        "sent_id": [0],
                    },
                },
            ]
            write_jsonl(queries_path, source_queries)

            baseline_records = [
                {
                    "query_id": "q_bool",
                    "query": source_queries[0]["question"],
                    "expected_answer": "yes",
                    "config_name": "baseline",
                    "supporting_facts": source_queries[0]["supporting_facts"],
                    "retrieved_chunks": [
                        {
                            "chunk_id": "chunk-1",
                            "doc_id": "wiki::Scott Derrickson",
                            "topic_id": "Scott Derrickson",
                            "rank": 1,
                            "score": 0.91,
                            "is_malicious": False,
                            "attack_type": None,
                        },
                        {
                            "chunk_id": "chunk-2",
                            "doc_id": "wiki::Ed Wood",
                            "topic_id": "Ed Wood",
                            "rank": 2,
                            "score": 0.88,
                            "is_malicious": False,
                            "attack_type": None,
                        },
                    ],
                    "model_output": "Yes, both people are American.",
                    "final_output": "Yes, both people are American.",
                    "support_coverage": {
                        "titles_in_corpus": {
                            "Scott Derrickson": True,
                            "Ed Wood": True,
                        },
                        "titles_retrieved": {
                            "Scott Derrickson": True,
                            "Ed Wood": True,
                        },
                    },
                    "defense_decisions": {"defenses_enabled": []},
                    "latencies_ms": {"generation": 1000.0},
                },
                {
                    "query_id": "q_role",
                    "query": source_queries[1]["question"],
                    "expected_answer": "Chief of Protocol",
                    "config_name": "baseline",
                    "supporting_facts": source_queries[1]["supporting_facts"],
                    "retrieved_chunks": [
                        {
                            "chunk_id": "chunk-3",
                            "doc_id": "wiki::Kiss and Tell (1945 film)",
                            "topic_id": "Kiss and Tell (1945 film)",
                            "rank": 1,
                            "score": 0.93,
                            "is_malicious": False,
                            "attack_type": None,
                        },
                        {
                            "chunk_id": "chunk-4",
                            "doc_id": "wiki::Shirley Temple",
                            "topic_id": "Shirley Temple",
                            "rank": 2,
                            "score": 0.87,
                            "is_malicious": False,
                            "attack_type": None,
                        },
                    ],
                    "model_output": "She later served as Chief of Protocol of the United States.",
                    "final_output": "She later served as Chief of Protocol of the United States.",
                    "support_coverage": {
                        "titles_in_corpus": {
                            "Kiss and Tell (1945 film)": True,
                            "Shirley Temple": True,
                        },
                        "titles_retrieved": {
                            "Kiss and Tell (1945 film)": True,
                            "Shirley Temple": False,
                        },
                    },
                    "defense_decisions": {"defenses_enabled": []},
                    "latencies_ms": {"generation": 1000.0},
                },
                {
                    "query_id": "q_year",
                    "query": source_queries[2]["question"],
                    "expected_answer": "1984",
                    "config_name": "baseline",
                    "supporting_facts": source_queries[2]["supporting_facts"],
                    "retrieved_chunks": [
                        {
                            "chunk_id": "chunk-5",
                            "doc_id": "wiki::Distractor",
                            "topic_id": "Distractor",
                            "rank": 1,
                            "score": 0.45,
                            "is_malicious": False,
                            "attack_type": None,
                        }
                    ],
                    "model_output": "The answer cannot be determined from the retrieved context.",
                    "final_output": "The answer cannot be determined from the retrieved context.",
                    "support_coverage": {
                        "titles_in_corpus": {"Sample Person": True},
                        "titles_retrieved": {"Sample Person": False},
                    },
                    "defense_decisions": {"defenses_enabled": []},
                    "latencies_ms": {"generation": 1000.0},
                },
            ]
            logs_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(logs_dir / "baseline_runs.jsonl", baseline_records)

            validate_close_ended_query_set(
                dataset="close_ended_dev",
                config_name="baseline",
                logs_dir=logs_dir,
                queries_path=queries_path,
                validated_queries_path=validated_queries_path,
                failed_queries_path=failed_queries_path,
                export_manifest_path=export_manifest_path,
                manual_review_path=manual_review_path,
            )

            validated_queries = [
                json.loads(line)
                for line in validated_queries_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            failed_queries = [
                json.loads(line)
                for line in failed_queries_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            manual_review_rows = [
                json.loads(line)
                for line in manual_review_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(
                [query["query_id"] for query in validated_queries],
                ["q_bool"],
            )
            self.assertEqual(
                [query["query_id"] for query in failed_queries],
                ["q_role", "q_year"],
            )
            self.assertEqual(
                [row["query_id"] for row in manual_review_rows],
                ["q_bool"],
            )
            self.assertEqual(export_manifest["validated_query_count"], 1)
            self.assertEqual(export_manifest["failed_query_count"], 2)


if __name__ == "__main__":
    unittest.main()
