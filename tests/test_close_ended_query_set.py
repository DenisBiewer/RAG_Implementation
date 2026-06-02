from __future__ import annotations

import json
import sys
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from scripts.data import build_close_ended_query_set as builder


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class TestCloseEndedQuerySetBuilder(unittest.TestCase):
    def test_build_close_ended_query_set_filters_and_writes_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            corpus_path = root / "corpus.jsonl"
            queries_path = root / "queries.jsonl"

            corpus_records = [
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
                {
                    "doc_id": "wiki::Kiss and Tell (1945 film)",
                    "text": "Kiss and Tell is a 1945 film starring Shirley Temple as Corliss Archer.",
                    "document_type": "wiki_article",
                    "topic_id": "Kiss and Tell (1945 film)",
                    "timestamp": None,
                    "is_malicious": False,
                    "attack_type": None,
                },
                {
                    "doc_id": "wiki::Shirley Temple",
                    "text": "Shirley Temple later served as Chief of Protocol of the United States.",
                    "document_type": "wiki_article",
                    "topic_id": "Shirley Temple",
                    "timestamp": None,
                    "is_malicious": False,
                    "attack_type": None,
                },
                {
                    "doc_id": "wiki::Explanatory Topic",
                    "text": "This article exists only to support a rejected query.",
                    "document_type": "wiki_article",
                    "topic_id": "Explanatory Topic",
                    "timestamp": None,
                    "is_malicious": False,
                    "attack_type": None,
                },
                {
                    "doc_id": "wiki::Complex Topic",
                    "text": "This article references a rejected punctuation-heavy answer.",
                    "document_type": "wiki_article",
                    "topic_id": "Complex Topic",
                    "timestamp": None,
                    "is_malicious": False,
                    "attack_type": None,
                },
                {
                    "doc_id": "wiki::Known Lake",
                    "text": "Known Lake is in Example County.",
                    "document_type": "wiki_article",
                    "topic_id": "Known Lake",
                    "timestamp": None,
                    "is_malicious": False,
                    "attack_type": None,
                },
                {
                    "doc_id": "wiki::Example County",
                    "text": "Example County is a county in Example State.",
                    "document_type": "wiki_article",
                    "topic_id": "Example County",
                    "timestamp": None,
                    "is_malicious": False,
                    "attack_type": None,
                },
                {
                    "doc_id": "wiki::Distractor",
                    "text": "Distractor article.",
                    "document_type": "wiki_article",
                    "topic_id": "Distractor",
                    "timestamp": None,
                    "is_malicious": False,
                    "attack_type": None,
                },
            ]
            query_records = [
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
                    "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
                    "answer": "Chief of Protocol",
                    "supporting_facts": {
                        "title": ["Kiss and Tell (1945 film)", "Shirley Temple"],
                        "sent_id": [0, 0],
                    },
                },
                {
                    "query_id": "q_long",
                    "question": "What did the explanatory topic say?",
                    "answer": "It explains the historical background in several detailed clauses",
                    "supporting_facts": {
                        "title": ["Explanatory Topic"],
                        "sent_id": [0],
                    },
                },
                {
                    "query_id": "q_punct",
                    "question": "What is the complex title?",
                    "answer": "Alpha/Beta; Gamma",
                    "supporting_facts": {
                        "title": ["Complex Topic"],
                        "sent_id": [0],
                    },
                },
                {
                    "query_id": "q_missing_support",
                    "question": "Where is the missing monument located?",
                    "answer": "Example City",
                    "supporting_facts": {
                        "title": ["Missing Monument"],
                        "sent_id": [0],
                    },
                },
                {
                    "query_id": "q_missing_answer",
                    "question": "Where is Known Lake located?",
                    "answer": "Example City",
                    "supporting_facts": {
                        "title": ["Known Lake", "Example County"],
                        "sent_id": [0, 0],
                    },
                },
            ]

            write_jsonl(corpus_path, corpus_records)
            write_jsonl(queries_path, query_records)

            output_dir = root / "close_ended_dev"
            builder.build_close_ended_query_set(
                dataset="close_ended_dev",
                corpus_path=corpus_path,
                queries_path=queries_path,
                output_corpus_path=output_dir / "corpus.jsonl",
                output_queries_path=output_dir / "queries.jsonl",
                accepted_audit_path=output_dir / "accepted_candidates.jsonl",
                rejected_audit_path=output_dir / "rejected_candidates.jsonl",
                manifest_path=output_dir / "manifest.json",
                initial_candidate_count=2,
                final_target_count=2,
                distractor_count=1,
                seed=42,
            )

            accepted_queries = [
                json.loads(line)
                for line in (output_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            accepted_audit = [
                json.loads(line)
                for line in (output_dir / "accepted_candidates.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rejected = [
                json.loads(line)
                for line in (output_dir / "rejected_candidates.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual([record["query_id"] for record in accepted_queries], ["q_bool", "q_role"])
            self.assertEqual(len(accepted_audit), 2)
            self.assertFalse(accepted_audit[0]["answer_found_in_support"])
            self.assertTrue(accepted_audit[1]["answer_found_in_support"])

            rejection_reasons = {record["query_id"]: record["rejection_reason"] for record in rejected}
            self.assertEqual(rejection_reasons["q_long"], "answer_too_long")
            self.assertEqual(rejection_reasons["q_punct"], "complex_answer_punctuation")
            self.assertEqual(rejection_reasons["q_missing_support"], "support_not_found_in_corpus")
            self.assertEqual(rejection_reasons["q_missing_answer"], "answer_not_found_in_support")

            self.assertEqual(manifest["accepted_question_count"], 2)
            self.assertEqual(manifest["rejected_question_count"], 4)
            self.assertEqual(manifest["corpus_record_count"], 5)
            self.assertEqual(manifest["query_selection"]["selected_query_ids"], ["q_bool", "q_role"])
            self.assertEqual(manifest["query_selection"]["method"], "stable_seeded_hash_rank")
            self.assertTrue((output_dir / "corpus.jsonl").exists())
            self.assertTrue((output_dir / "queries.jsonl").exists())
            self.assertTrue((output_dir / "accepted_candidates.jsonl").exists())
            self.assertTrue((output_dir / "rejected_candidates.jsonl").exists())
            self.assertTrue((output_dir / "manifest.json").exists())

    def test_main_entry_point_runs_with_cli_arguments(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            corpus_path = root / "corpus.jsonl"
            queries_path = root / "queries.jsonl"

            write_jsonl(
                corpus_path,
                [
                    {
                        "doc_id": "wiki::Sample Person",
                        "text": "Sample Person was born in 1984.",
                        "document_type": "wiki_article",
                        "topic_id": "Sample Person",
                        "timestamp": None,
                        "is_malicious": False,
                        "attack_type": None,
                    },
                    {
                        "doc_id": "wiki::Sample Film",
                        "text": "Sample Film starred Sample Person.",
                        "document_type": "wiki_article",
                        "topic_id": "Sample Film",
                        "timestamp": None,
                        "is_malicious": False,
                        "attack_type": None,
                    },
                ],
            )
            write_jsonl(
                queries_path,
                [
                    {
                        "query_id": "q1",
                        "question": "In what year was the star of Sample Film born?",
                        "answer": "1984",
                        "supporting_facts": {
                            "title": ["Sample Film", "Sample Person"],
                            "sent_id": [0, 0],
                        },
                    }
                ],
            )

            output_dir = root / "cli_output"
            argv = [
                "build_close_ended_query_set.py",
                "--corpus-path",
                str(corpus_path),
                "--queries-path",
                str(queries_path),
                "--output-corpus-path",
                str(output_dir / "corpus.jsonl"),
                "--output-queries-path",
                str(output_dir / "queries.jsonl"),
                "--accepted-audit-path",
                str(output_dir / "accepted_candidates.jsonl"),
                "--rejected-audit-path",
                str(output_dir / "rejected_candidates.jsonl"),
                "--manifest-path",
                str(output_dir / "manifest.json"),
                "--query-count",
                "1",
                "--distractor-count",
                "0",
                "--seed",
                "42",
            ]

            with patch.object(sys, "argv", argv):
                builder.main()

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["accepted_question_count"], 1)
            self.assertEqual(manifest["rejected_question_count"], 0)
            self.assertTrue((output_dir / "queries.jsonl").exists())
            self.assertTrue((output_dir / "corpus.jsonl").exists())

    def test_close_ended_main_uses_full_corpus_but_rejects_missing_support_titles(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            corpus_path = root / "corpus.jsonl"
            queries_path = root / "queries.jsonl"

            write_jsonl(
                corpus_path,
                [
                    {
                        "doc_id": "wiki::Doc A",
                        "text": "Doc A says the answer is yes.",
                        "document_type": "wiki_article",
                        "topic_id": "Doc A",
                        "timestamp": None,
                        "is_malicious": False,
                        "attack_type": None,
                    },
                    {
                        "doc_id": "wiki::Doc B",
                        "text": "Doc B is unrelated.",
                        "document_type": "wiki_article",
                        "topic_id": "Doc B",
                        "timestamp": None,
                        "is_malicious": False,
                        "attack_type": None,
                    },
                    {
                        "doc_id": "wiki::Doc C",
                        "text": "Doc C confirms the valid event happened in 1999.",
                        "document_type": "wiki_article",
                        "topic_id": "Doc C",
                        "timestamp": None,
                        "is_malicious": False,
                        "attack_type": None,
                    },
                ],
            )
            write_jsonl(
                queries_path,
                [
                    {
                        "query_id": "q1",
                        "question": "Was the archived statement confirmed?",
                        "answer": "yes",
                        "supporting_facts": {
                            "title": ["Missing Support Title"],
                            "sent_id": [0],
                        },
                    },
                    {
                        "query_id": "q2",
                        "question": "In what year did the valid event happen?",
                        "answer": "1999",
                        "supporting_facts": {
                            "title": ["Doc C"],
                            "sent_id": [0],
                        },
                    },
                ],
            )

            output_dir = root / "close_ended_main"
            builder.build_close_ended_query_set(
                dataset="close_ended_main",
                corpus_path=corpus_path,
                queries_path=queries_path,
                output_corpus_path=output_dir / "corpus.jsonl",
                output_queries_path=output_dir / "queries.jsonl",
                accepted_audit_path=output_dir / "accepted_candidates.jsonl",
                rejected_audit_path=output_dir / "rejected_candidates.jsonl",
                manifest_path=output_dir / "manifest.json",
                initial_candidate_count=1,
                final_target_count=20,
                distractor_count=0,
                seed=7,
            )

            accepted_audit = [
                json.loads(line)
                for line in (output_dir / "accepted_candidates.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rejected_audit = [
                json.loads(line)
                for line in (output_dir / "rejected_candidates.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            copied_corpus = (output_dir / "corpus.jsonl").read_text(encoding="utf-8")
            source_corpus = corpus_path.read_text(encoding="utf-8")
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(len(accepted_audit), 1)
            self.assertEqual(accepted_audit[0]["query_id"], "q2")
            self.assertTrue(accepted_audit[0]["all_support_titles_present_in_corpus"])
            self.assertEqual(len(rejected_audit), 1)
            self.assertEqual(rejected_audit[0]["query_id"], "q1")
            self.assertEqual(rejected_audit[0]["rejection_reason"], "support_not_found_in_corpus")
            self.assertEqual(copied_corpus, source_corpus)
            self.assertTrue(manifest["use_full_corpus"])
            self.assertTrue(manifest["support_title_inclusion_required"])
            self.assertEqual(manifest["accepted_question_count"], 1)
            self.assertEqual(manifest["distractor_selection"]["method"], "full_corpus_copy")


if __name__ == "__main__":
    unittest.main()
