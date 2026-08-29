from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path

from scripts.analysis.analyze_clean_dataset import (
    analyze_clean_dataset,
    answer_matches,
)
from scripts.indexing.indexing_presets import INDEXING_PRESETS
from scripts.run.run_experiments import DATASET_PROFILES, apply_log_variant
from app.dataset_registry import format_temperature_variant


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class TestValidationMatching(unittest.TestCase):
    def test_answer_matches_handles_boolean_and_span_outputs(self):
        strict_match, lenient_match = answer_matches(
            "Yes, both people are American.",
            "yes",
        )
        self.assertTrue(strict_match)
        self.assertTrue(lenient_match)

        strict_match, lenient_match = answer_matches(
            "No, they were not of the same nationality.",
            "yes",
        )
        self.assertFalse(strict_match)
        self.assertFalse(lenient_match)

        strict_match, lenient_match = answer_matches(
            "She later served as Chief of Protocol of the United States.",
            "Chief of Protocol",
        )
        self.assertTrue(strict_match)
        self.assertTrue(lenient_match)


class TestCleanBaselineValidationAnalyzer(unittest.TestCase):
    def test_analyze_clean_dataset_writes_analysis_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queries_path = root / "queries.jsonl"
            logs_dir = root / "logs"
            results_dir = root / "results"

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

            analyze_clean_dataset(
                dataset="close_ended_dev",
                config_names=["baseline"],
                logs_dir=logs_dir,
                queries_path=queries_path,
                results_dir=results_dir,
            )

            summary = json.loads(
                (results_dir / "validation" / "close_ended_dev" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            per_query_rows = [
                json.loads(line)
                for line in (
                    results_dir / "validation" / "close_ended_dev" / "per_query_results.jsonl"
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            analysis_manual_review_rows = [
                json.loads(line)
                for line in (
                    results_dir / "validation" / "close_ended_dev" / "manual_review.jsonl"
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            baseline_summary = summary["summary_by_config"]["baseline"]
            self.assertEqual(baseline_summary["total_records"], 3)
            self.assertEqual(baseline_summary["strict_rule_answer_matches"], 2)
            self.assertEqual(baseline_summary["all_support_titles_retrieved"], 1)
            self.assertEqual(baseline_summary["validation_passes"], 1)
            self.assertAlmostEqual(baseline_summary["strict_rule_answer_match_rate"], 0.6667)

            by_query_id = {row["query_id"]: row for row in per_query_rows}
            self.assertTrue(by_query_id["q_bool"]["validation_pass"])
            self.assertFalse(by_query_id["q_role"]["validation_pass"])
            self.assertFalse(by_query_id["q_year"]["validation_pass"])
            self.assertFalse(by_query_id["q_role"]["all_support_titles_retrieved"])

            self.assertEqual(len(analysis_manual_review_rows), 2)
            manual_review_by_query_id = {
                row["query_id"]: row
                for row in analysis_manual_review_rows
            }
            self.assertIn("q_year", manual_review_by_query_id)
            self.assertEqual(manual_review_by_query_id["q_year"]["attacker_target_answer"], None)
            self.assertIn(
                "refusal",
                manual_review_by_query_id["q_year"]["manual_review_reasons"],
            )
            self.assertTrue(
                any(
                    "lenient_match_audit_sample" in row["manual_review_reasons"]
                    for row in analysis_manual_review_rows
                )
            )

    def test_close_ended_dataset_profiles_and_presets_cover_dev_and_main(self):
        self.assertIn("close_ended_dev_initial", INDEXING_PRESETS)
        self.assertIn("close_ended_dev_final", INDEXING_PRESETS)
        self.assertNotIn("close_ended_main_initial", INDEXING_PRESETS)
        self.assertNotIn("close_ended_main_final", INDEXING_PRESETS)
        self.assertIn("close_ended_dev_initial", DATASET_PROFILES)
        self.assertIn("close_ended_dev_final", DATASET_PROFILES)
        self.assertIn("close_ended_main_initial", DATASET_PROFILES)
        self.assertIn("close_ended_main_final", DATASET_PROFILES)

    def test_temperature_variant_paths_use_nested_subdirectories(self):
        variant = format_temperature_variant(0.7)
        profile = apply_log_variant(DATASET_PROFILES["close_ended_main_final"], variant=variant)

        self.assertEqual(variant, "temperature_0_7")
        self.assertTrue(str(profile.log_dir).endswith("close_ended_main\\final\\temperature_0_7"))

    def test_analyze_clean_dataset_writes_temperature_variant_results_separately(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queries_path = root / "queries.jsonl"
            logs_dir = root / "logs" / "temperature_0_7"
            results_dir = root / "results"
            variant = "temperature_0_7"

            source_queries = [
                {
                    "query_id": "q1",
                    "question": "Who wrote notes about computation?",
                    "answer": "Ada Lovelace",
                    "supporting_facts": {"title": ["Ada Lovelace"], "sent_id": [0]},
                }
            ]
            write_jsonl(queries_path, source_queries)
            logs_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(
                logs_dir / "baseline_runs.jsonl",
                [
                    {
                        "query_id": "q1",
                        "query": source_queries[0]["question"],
                        "expected_answer": "Ada Lovelace",
                        "config_name": "baseline",
                        "supporting_facts": source_queries[0]["supporting_facts"],
                        "retrieved_chunks": [],
                        "model_output": "Ada Lovelace",
                        "final_output": "Ada Lovelace",
                        "support_coverage": {
                            "titles_in_corpus": {"Ada Lovelace": True},
                            "titles_retrieved": {"Ada Lovelace": True},
                        },
                        "defense_decisions": {"defenses_enabled": []},
                        "latencies_ms": {"generation": 1000.0},
                    }
                ],
            )

            analyze_clean_dataset(
                dataset="close_ended_dev",
                config_names=["baseline"],
                logs_dir=logs_dir,
                queries_path=queries_path,
                results_dir=results_dir,
                variant=variant,
            )

            self.assertTrue(
                (results_dir / "validation" / "close_ended_dev" / variant / "summary.json").exists()
            )
            self.assertFalse(
                (results_dir / "validation" / "close_ended_dev" / "summary.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
