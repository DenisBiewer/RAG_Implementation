"""
Unit tests for `app.vector_index.FaissIndex`.

`TestFaissIndex` verifies the wrapper logic implemented in `FaissIndex`, namely:
- `add` rejects mismatched embedding/chunk counts.
- `add` converts inputs to contiguous `float32` arrays before passing them to FAISS.
- `search` accepts a single query vector, caps `top_k`, returns empty arrays for an
  empty index, and rejects invalid `top_k` values.
- `save` and `load` preserve the stored vectors and row-to-chunk metadata.
- `load` validates that metadata rows still match the persisted index rows.
- `load` uses `cls`, so subclass calls return an instance of the subclass.

These tests exercise the real `FaissIndex` class while replacing `faiss` and
`app.schemas.Chunk` with small test doubles so the behavior of the wrapper can be
checked in isolation from external/runtime dependencies.
"""

import importlib
import json
import sys
import tempfile
import types
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _build_fake_schemas_module() -> types.ModuleType:
    module = types.ModuleType("app.schemas")

    @dataclass
    class Chunk:
        chunk_id: str
        doc_id: str
        text: str
        token_start: int
        token_end: int
        document_type: str
        topic_id: str
        timestamp: str | None = None
        is_malicious: bool = False
        attack_type: str | None = None

        def model_dump(self) -> dict:
            return asdict(self)

    module.Chunk = Chunk
    return module


def _build_fake_faiss_module() -> types.ModuleType:
    module = types.ModuleType("faiss")
    stored_indices: dict[str, dict[str, np.ndarray | int]] = {}

    class FakeIndexFlatIP:
        def __init__(self, dim: int):
            self.d = dim
            self.ntotal = 0
            self.vectors = np.empty((0, dim), dtype=np.float32)
            self.last_added: np.ndarray | None = None

        def add(self, embeddings: np.ndarray) -> None:
            self.last_added = embeddings
            self.vectors = np.vstack([self.vectors, embeddings])
            self.ntotal = self.vectors.shape[0]

        def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
            scores = query @ self.vectors.T
            ranked = np.argsort(-scores, axis=1)[:, :top_k]
            top_scores = np.take_along_axis(scores, ranked, axis=1)
            return top_scores.astype(np.float32), ranked.astype(np.int64)

    def write_index(index: FakeIndexFlatIP, path: str) -> None:
        stored_indices[path] = {
            "d": index.d,
            "vectors": index.vectors.copy(),
            "ntotal": index.ntotal,
        }

    def read_index(path: str) -> FakeIndexFlatIP:
        payload = stored_indices[path]
        restored = FakeIndexFlatIP(int(payload["d"]))
        restored.vectors = payload["vectors"].copy()
        restored.ntotal = int(payload["ntotal"])
        return restored

    module.IndexFlatIP = FakeIndexFlatIP
    module.write_index = write_index
    module.read_index = read_index
    return module


class TestFaissIndex(unittest.TestCase):
    def setUp(self) -> None:
        sys.modules.pop("app.vector_index", None)
        self.fake_faiss = _build_fake_faiss_module()
        self.fake_schemas = _build_fake_schemas_module()
        self.module_patcher = patch.dict(
            sys.modules,
            {
                "faiss": self.fake_faiss,
                "app.schemas": self.fake_schemas,
            },
        )
        self.module_patcher.start()
        self.vector_index = importlib.import_module("app.vector_index")
        self.FaissIndex = self.vector_index.FaissIndex
        self.Chunk = self.fake_schemas.Chunk

    def tearDown(self) -> None:
        self.module_patcher.stop()
        sys.modules.pop("app.vector_index", None)

    def _chunk(self, chunk_id: str, text: str, *, topic_id: str = "topic-a"):
        return self.Chunk(
            chunk_id=chunk_id,
            doc_id=f"doc::{chunk_id}",
            text=text,
            token_start=0,
            token_end=len(text.split()),
            document_type="wiki_chunk",
            topic_id=topic_id,
        )

    def test_add_rejects_mismatched_row_count(self):
        index = self.FaissIndex(dim=2)
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        chunks = [self._chunk("c1", "Ada"), self._chunk("c2", "Lovelace")]

        with self.assertRaisesRegex(ValueError, "Number of embeddings must match"):
            index.add(embeddings, chunks)

    def test_add_converts_embeddings_to_contiguous_float32(self):
        index = self.FaissIndex(dim=2)
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64, order="F")

        index.add(embeddings, [self._chunk("c1", "Ada"), self._chunk("c2", "Lovelace")])

        added = index.index.last_added
        self.assertEqual(added.dtype, np.float32)
        self.assertTrue(added.flags["C_CONTIGUOUS"])
        self.assertEqual(index.row_to_chunk[0]["chunk_id"], "c1")
        self.assertEqual(index.row_to_chunk[1]["chunk_id"], "c2")

    def test_search_accepts_single_vector_and_caps_top_k(self):
        index = self.FaissIndex(dim=2)
        index.add(
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            [self._chunk("c1", "Ada"), self._chunk("c2", "Lovelace")],
        )

        scores, indices = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=5)

        self.assertEqual(scores.shape, (2,))
        self.assertEqual(indices.shape, (2,))
        self.assertEqual(indices.tolist(), [0, 1])

    def test_search_returns_empty_arrays_for_empty_index(self):
        index = self.FaissIndex(dim=2)

        scores, indices = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)

        self.assertEqual(scores.tolist(), [])
        self.assertEqual(indices.tolist(), [])

    def test_search_rejects_invalid_top_k(self):
        index = self.FaissIndex(dim=2)

        with self.assertRaisesRegex(ValueError, "top_k must be a positive integer"):
            index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=0)

    def test_save_and_load_round_trip_preserves_index_and_metadata(self):
        index = self.FaissIndex(dim=2)
        index.add(
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            [self._chunk("c1", "Ada"), self._chunk("c2", "Lovelace")],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = Path(tmp_dir) / "index.faiss"
            meta_path = Path(tmp_dir) / "index.json"

            index.save(index_path, meta_path)
            loaded = self.FaissIndex.load(index_path, meta_path)

        self.assertEqual(loaded.index.d, 2)
        self.assertEqual(loaded.index.ntotal, 2)
        self.assertEqual(loaded.row_to_chunk, index.row_to_chunk)

    def test_load_uses_cls_and_returns_subclass_instance(self):
        class CustomFaissIndex(self.FaissIndex):
            pass

        index = CustomFaissIndex(dim=2)
        index.add(
            np.array([[1.0, 0.0]], dtype=np.float32),
            [self._chunk("c1", "Ada")],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = Path(tmp_dir) / "index.faiss"
            meta_path = Path(tmp_dir) / "index.json"
            index.save(index_path, meta_path)
            loaded = CustomFaissIndex.load(index_path, meta_path)

        self.assertIsInstance(loaded, CustomFaissIndex)

    def test_load_rejects_metadata_that_does_not_match_index_rows(self):
        index = self.FaissIndex(dim=2)
        index.add(
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            [self._chunk("c1", "Ada"), self._chunk("c2", "Lovelace")],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = Path(tmp_dir) / "index.faiss"
            meta_path = Path(tmp_dir) / "index.json"
            index.save(index_path, meta_path)

            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump({"0": index.row_to_chunk[0]}, handle)

            with self.assertRaisesRegex(ValueError, "Loaded metadata rows do not match"):
                self.FaissIndex.load(index_path, meta_path)


if __name__ == "__main__":
    unittest.main()
