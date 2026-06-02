import unittest

from app.experiment_runner import AVAILABLE_CONFIGS, resolve_run_configs


class TestExperimentConfigs(unittest.TestCase):
    def test_single_defense_configs_are_available(self):
        expected_configs = {
            "mmr_only",
            "retrieval_filtering_only",
            "prompt_isolation_only",
            "instruction_hierarchy_only",
            "grounding_only",
            "prompt_context_flags_only",
        }

        self.assertTrue(expected_configs.issubset(set(AVAILABLE_CONFIGS)))

    def test_resolve_run_configs_supports_single_defense_ablation(self):
        resolved = resolve_run_configs(
            ["retrieval_filtering_only", "prompt_context_flags_only"]
        )

        self.assertEqual(
            [config.run_name for config in resolved],
            ["retrieval_filtering_only", "prompt_context_flags_only"],
        )


if __name__ == "__main__":
    unittest.main()
