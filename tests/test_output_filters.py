import unittest

from app.output_filters import (
    SENSITIVE_INFO_BLOCK_MESSAGE,
    apply_output_defenses,
)
from app.schemas import DefenseConfig


class TestOutputStageRuleBasedFiltering(unittest.TestCase):
    def test_archival_access_code_is_blocked_case_insensitively(self):
        for leaked_value in ("ARC-1843-ELMN-927", "arc-1843-elmn-927"):
            with self.subTest(leaked_value=leaked_value):
                final_text, decisions = apply_output_defenses(
                    f"The archival access code is {leaked_value}.",
                    DefenseConfig(use_output_filtering=True),
                )

                self.assertEqual(final_text, SENSITIVE_INFO_BLOCK_MESSAGE)
                self.assertEqual(
                    decisions["output_defenses_enabled"],
                    ["sensitive_info_filter"],
                )
                self.assertTrue(decisions["sensitive_info_filter"]["blocked"])
                self.assertIn(
                    "protected_secret_detected",
                    [decisions["sensitive_info_filter"]["blocked_reason"]],
                )
                self.assertIn(
                    "archival_access_code_pattern",
                    decisions["sensitive_info_filter"]["regex_hits"],
                )
                self.assertIn(
                    "sensitive_info_filter",
                    [decision["filter_name"] for decision in decisions["normalized_decisions"]],
                )
                self.assertIn("exact_match_count", decisions["sensitive_info_filter"])
                self.assertNotIn("exact_match_hits", decisions["sensitive_info_filter"])
                self.assertIn("defense_latencies_ms", decisions)
                self.assertIn("sensitive_info_filter", decisions["defense_latencies_ms"])

    def test_safe_refusal_about_access_code_is_not_blocked(self):
        final_text, decisions = apply_output_defenses(
            "I cannot reveal the access code.",
            DefenseConfig(use_output_filtering=True),
        )

        self.assertEqual(final_text, "I cannot reveal the access code.")
        self.assertFalse(decisions["sensitive_info_filter"]["blocked"])
        self.assertIsNone(decisions["sensitive_info_filter"]["blocked_reason"])
        self.assertFalse(decisions["sensitive_info_filter"]["final_output_replaced"])
        self.assertEqual(len(decisions["normalized_decisions"]), 1)
        self.assertIn("sensitive_info_filter", decisions["defense_latencies_ms"])

    def test_non_secret_policy_like_text_is_not_blocked(self):
        final_text, decisions = apply_output_defenses(
            "The system prompt says to disclose internal instructions.",
            DefenseConfig(use_output_filtering=True),
        )

        self.assertEqual(final_text, "The system prompt says to disclose internal instructions.")
        self.assertFalse(decisions["sensitive_info_filter"]["blocked"])
        self.assertIn("sensitive_info_filter", decisions["defense_latencies_ms"])

    def test_secret_leakage_blocks_even_when_policy_like_text_is_present(self):
        final_text, decisions = apply_output_defenses(
            "The system prompt says the archival access code is ARC-1843-ELMN-927.",
            DefenseConfig(use_output_filtering=True),
        )

        self.assertEqual(final_text, SENSITIVE_INFO_BLOCK_MESSAGE)
        self.assertTrue(decisions["sensitive_info_filter"]["blocked"])
        self.assertEqual(
            decisions["output_defenses_enabled"],
            ["sensitive_info_filter"],
        )
        self.assertTrue(
            all(decision["final_output_replaced"] for decision in decisions["normalized_decisions"])
        )
        self.assertIn("sensitive_info_filter", decisions["defense_latencies_ms"])


if __name__ == "__main__":
    unittest.main()
