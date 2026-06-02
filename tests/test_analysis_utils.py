import unittest

from app.analysis_utils import (
    compact_retrieved_chunk,
    detector_metrics_by_config,
    finalize_detector_metrics,
    fraction,
    manual_review_chunk_summary,
    update_detector_metrics,
)


class TestAnalysisUtils(unittest.TestCase):
    def test_fraction_is_zero_safe_and_rounded(self):
        self.assertEqual(fraction(0, 0), 0.0)
        self.assertEqual(fraction(1, 3), 0.3333)

    def test_compact_retrieved_chunk_keeps_shared_fields(self):
        chunk = {
            "chunk_id": "chunk-1",
            "doc_id": "doc-1",
            "topic_id": "topic-1",
            "rank": 2,
            "score": 0.81234567,
            "is_malicious": True,
            "attack_type": "instruction_following",
            "extra_field": "ignored",
        }

        compact = compact_retrieved_chunk(chunk)

        self.assertEqual(
            compact,
            {
                "chunk_id": "chunk-1",
                "doc_id": "doc-1",
                "topic_id": "topic-1",
                "rank": 2,
                "score": 0.812346,
                "is_malicious": True,
                "attack_type": "instruction_following",
            },
        )

    def test_manual_review_chunk_summary_truncates_preview(self):
        chunk = {
            "chunk_id": "chunk-1",
            "doc_id": "wiki::Scott Derrickson",
            "topic_id": "Scott Derrickson",
            "rank": 1,
            "is_malicious": False,
            "attack_type": None,
            "text": "x" * 305,
        }

        summary = manual_review_chunk_summary(chunk)

        self.assertEqual(summary["title"], "Scott Derrickson")
        self.assertEqual(summary["rank"], 1)
        self.assertTrue(summary["text_preview"].endswith("..."))

    def test_detector_metric_helpers_accumulate_and_finalize(self):
        metrics: dict[str, dict] = {}
        decision_log = [
            {
                "detector_name": "retrieval_filtering",
                "stage": "retrieval",
                "is_malicious": True,
                "flagged": True,
            },
            {
                "detector_name": "retrieval_filtering",
                "stage": "retrieval",
                "is_malicious": False,
                "flagged": False,
            },
            {
                "filter_name": "sensitive_info_filter",
                "stage": "output",
                "blocked": True,
            },
            {
                "filter_name": "sensitive_info_filter",
                "stage": "output",
                "blocked": False,
            },
        ]

        update_detector_metrics(
            metrics,
            decision_log,
            is_attack_output=True,
            is_benign_output=False,
        )
        finalized = finalize_detector_metrics(metrics)

        self.assertEqual(finalized["retrieval_filtering"]["malicious_inspected"], 1)
        self.assertEqual(finalized["retrieval_filtering"]["benign_inspected"], 1)
        self.assertEqual(finalized["retrieval_filtering"]["malicious_flagged"], 1)
        self.assertEqual(finalized["retrieval_filtering"]["benign_flagged"], 0)
        self.assertEqual(finalized["retrieval_filtering"]["hit_rate"], 1.0)
        self.assertEqual(finalized["retrieval_filtering"]["false_positive_rate"], 0.0)
        self.assertEqual(finalized["sensitive_info_filter"]["outputs_inspected"], 2)
        self.assertEqual(finalized["sensitive_info_filter"]["outputs_blocked"], 1)
        self.assertEqual(finalized["sensitive_info_filter"]["attack_outputs_inspected"], 2)
        self.assertEqual(finalized["sensitive_info_filter"]["attack_outputs_blocked"], 1)
        self.assertEqual(finalized["sensitive_info_filter"]["block_rate"], 0.5)

    def test_detector_metrics_by_config_extracts_nested_summaries(self):
        summary_by_config = {
            "baseline": {"detector_metrics": {"sensitive_info_filter": {"stage": "output"}}},
            "mmr_only": {"status": "ok"},
        }

        extracted = detector_metrics_by_config(summary_by_config)

        self.assertEqual(
            extracted,
            {
                "baseline": {"sensitive_info_filter": {"stage": "output"}},
                "mmr_only": {},
            },
        )


if __name__ == "__main__":
    unittest.main()
