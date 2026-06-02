import unittest

from app.path_utils import BUILT_DIR, PROCESSED_DIR
from scripts.indexing.indexing_presets import (
    INDEXING_PRESETS,
    is_attack_preset,
    resolve_attack_base_paths,
    resolve_chunk_paths,
    resolve_index_paths,
    resolve_retrieval_artifact_paths,
)


class TestIndexingPresets(unittest.TestCase):
    def test_close_ended_dev_final_preset_has_consistent_paths(self):
        preset = INDEXING_PRESETS["close_ended_dev_final"]

        self.assertEqual(
            preset["input_path"],
            PROCESSED_DIR / "clean" / "close_ended_dev" / "final" / "corpus.jsonl",
        )
        self.assertEqual(
            preset["chunks_path"],
            BUILT_DIR / "clean" / "close_ended_dev" / "final" / "chunks.jsonl",
        )
        self.assertEqual(
            preset["index_path"],
            BUILT_DIR / "clean" / "close_ended_dev" / "final" / "faiss.index",
        )
        self.assertEqual(
            preset["meta_path"],
            BUILT_DIR / "clean" / "close_ended_dev" / "final" / "faiss_meta.json",
        )

    def test_resolve_helpers_share_one_source_of_truth(self):
        chunk_input_path, chunk_output_path = resolve_chunk_paths("instruction_following_close_ended_dev")
        index_chunks_path, index_path, meta_path = resolve_index_paths("instruction_following_close_ended_dev")
        artifact_paths = resolve_retrieval_artifact_paths("instruction_following_close_ended_dev")

        self.assertEqual(chunk_output_path, index_chunks_path)
        self.assertEqual(
            artifact_paths,
            (chunk_input_path, chunk_output_path, index_path, meta_path),
        )

    def test_exfiltration_close_ended_dev_control_preset_has_consistent_paths(self):
        preset = INDEXING_PRESETS["exfiltration_close_ended_dev_control"]

        self.assertEqual(
            preset["input_path"],
            PROCESSED_DIR / "attacks" / "exfiltration" / "close_ended_dev_control" / "corpus.jsonl",
        )
        self.assertEqual(
            preset["chunks_path"],
            BUILT_DIR / "attacks" / "exfiltration" / "close_ended_dev_control" / "chunks.jsonl",
        )
        self.assertEqual(
            preset["index_path"],
            BUILT_DIR / "attacks" / "exfiltration" / "close_ended_dev_control" / "faiss.index",
        )
        self.assertEqual(
            preset["meta_path"],
            BUILT_DIR / "attacks" / "exfiltration" / "close_ended_dev_control" / "faiss_meta.json",
        )

    def test_main_clean_reuses_complete_index_and_only_attack_indexes_are_built(self):
        self.assertNotIn("close_ended_main_initial", INDEXING_PRESETS)
        self.assertNotIn("close_ended_main_final", INDEXING_PRESETS)
        self.assertIn("instruction_following_close_ended_main", INDEXING_PRESETS)
        self.assertIn("exfiltration_close_ended_main", INDEXING_PRESETS)
        self.assertIn("exfiltration_close_ended_main_control", INDEXING_PRESETS)
        self.assertIn("poisoned_rag_close_ended_main", INDEXING_PRESETS)

    def test_attack_presets_resolve_the_expected_clean_retrieval_base(self):
        self.assertTrue(is_attack_preset("instruction_following_close_ended_main"))
        self.assertTrue(is_attack_preset("instruction_following_close_ended_dev"))

        main_base = resolve_attack_base_paths("instruction_following_close_ended_main")
        dev_base = resolve_attack_base_paths("instruction_following_close_ended_dev")

        self.assertEqual(main_base[0], "clean_complete")
        self.assertEqual(
            main_base[1],
            PROCESSED_DIR / "clean" / "complete" / "corpus.jsonl",
        )
        self.assertEqual(
            main_base[2],
            BUILT_DIR / "clean" / "complete" / "chunks.jsonl",
        )

        self.assertEqual(dev_base[0], "close_ended_dev_final")
        self.assertEqual(
            dev_base[1],
            PROCESSED_DIR / "clean" / "close_ended_dev" / "final" / "corpus.jsonl",
        )
        self.assertEqual(
            dev_base[2],
            BUILT_DIR / "clean" / "close_ended_dev" / "final" / "chunks.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
