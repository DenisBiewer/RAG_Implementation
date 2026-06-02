from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path

from scripts.indexing.attack_artifact_utils import (
    extract_appended_records,
    validate_base_build_settings,
    ensure_attack_output_paths_distinct,
    write_appended_jsonl,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class TestAttackArtifactUtils(unittest.TestCase):
    def test_extract_appended_records_returns_only_suffix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_path = root / "clean.jsonl"
            attacked_path = root / "attack.jsonl"

            clean_records = [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}]
            appended_records = [{"doc_id": "attack-1"}, {"doc_id": "attack-2"}]
            write_jsonl(base_path, clean_records)
            write_jsonl(attacked_path, clean_records + appended_records)

            clean_count, suffix = extract_appended_records(
                base_path=base_path,
                attacked_path=attacked_path,
            )

            self.assertEqual(clean_count, 2)
            self.assertEqual(suffix, appended_records)

    def test_extract_appended_records_rejects_prefix_divergence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_path = root / "clean.jsonl"
            attacked_path = root / "attack.jsonl"

            write_jsonl(base_path, [{"doc_id": "doc-1"}])
            write_jsonl(attacked_path, [{"doc_id": "different-doc"}])

            with self.assertRaisesRegex(ValueError, "diverges from the clean retrieval base"):
                extract_appended_records(base_path=base_path, attacked_path=attacked_path)

    def test_validate_base_build_settings_rejects_mismatch(self):
        with self.assertRaisesRegex(ValueError, "compatibility settings do not match"):
            validate_base_build_settings(
                {
                    "tokenizer_name": "base-tokenizer",
                    "chunk_size": 256,
                    "overlap": 32,
                },
                tokenizer_name="different-tokenizer",
                chunk_size=256,
                overlap=32,
            )

    def test_write_appended_jsonl_preserves_prefix_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prefix_path = root / "prefix.jsonl"
            output_path = root / "output.jsonl"

            prefix_records = [{"chunk_id": "clean-1"}, {"chunk_id": "clean-2"}]
            suffix_records = [{"chunk_id": "attack-1"}]
            write_jsonl(prefix_path, prefix_records)

            write_appended_jsonl(
                prefix_path=prefix_path,
                output_path=output_path,
                suffix_records=suffix_records,
            )

            merged = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(merged, prefix_records + suffix_records)

    def test_ensure_attack_output_paths_distinct_rejects_clean_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_chunks_path = root / "clean" / "chunks.jsonl"
            base_index_path = root / "clean" / "faiss.index"
            base_meta_path = root / "clean" / "faiss_meta.json"

            with self.assertRaisesRegex(ValueError, "would overwrite the clean retrieval base"):
                ensure_attack_output_paths_distinct(
                    base_chunks_path=base_chunks_path,
                    base_index_path=base_index_path,
                    base_meta_path=base_meta_path,
                    chunks_path=base_chunks_path,
                    index_path=root / "attack" / "faiss.index",
                    meta_path=root / "attack" / "faiss_meta.json",
                )


if __name__ == "__main__":
    unittest.main()
