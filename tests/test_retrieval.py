import unittest
from unittest.mock import MagicMock

import numpy as np

from app.retrieval import retrieve_top_k
from app.schemas import RetrievedChunk


class TestRetrieveTopK(unittest.TestCase):
    def test_retrieve_top_k_maps_hits_into_retrieved_chunks(self):
        index = MagicMock()
        index.search.return_value = (
            np.array([0.95, 0.75], dtype=np.float32),
            np.array([1, 0], dtype=np.int64),
        )
        index.row_to_chunk = {
            0: {
                "chunk_id": "c0",
                "doc_id": "doc::0",
                "text": "Ada wrote notes.",
                "is_malicious": False,
                "attack_type": None,
                "document_type": "wiki_chunk",
                "topic_id": "Ada Lovelace",
            },
            1: {
                "chunk_id": "c1",
                "doc_id": "doc::1",
                "text": "Charles Babbage built engines.",
                "is_malicious": True,
                "attack_type": "poisoning",
                "document_type": "wiki_chunk",
                "topic_id": "Charles Babbage",
            },
        }

        results = retrieve_top_k(
            index=index,
            query_embedding=np.array([1.0, 0.0], dtype=np.float32),
            top_k=2,
        )

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], RetrievedChunk)
        self.assertEqual(results[0].chunk_id, "c1")
        self.assertEqual(results[0].rank, 1)
        self.assertAlmostEqual(results[0].score, 0.95)
        self.assertTrue(results[0].is_malicious)
        self.assertEqual(results[0].attack_type, "poisoning")
        self.assertEqual(results[1].chunk_id, "c0")
        self.assertEqual(results[1].rank, 2)

    def test_retrieve_top_k_skips_negative_indices(self):
        index = MagicMock()
        index.search.return_value = (
            np.array([0.95, 0.0], dtype=np.float32),
            np.array([0, -1], dtype=np.int64),
        )
        index.row_to_chunk = {
            0: {
                "chunk_id": "c0",
                "doc_id": "doc::0",
                "text": "Ada wrote notes.",
                "is_malicious": False,
                "attack_type": None,
                "document_type": "wiki_chunk",
                "topic_id": "Ada Lovelace",
            }
        }

        results = retrieve_top_k(index, np.array([1.0, 0.0]), top_k=2)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "c0")
        self.assertEqual(results[0].rank, 1)

    def test_retrieve_top_k_passes_query_and_top_k_to_index(self):
        index = MagicMock()
        index.search.return_value = (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
        )
        query = np.array([0.2, 0.8], dtype=np.float32)

        results = retrieve_top_k(index, query, top_k=7)

        self.assertEqual(results, [])
        index.search.assert_called_once()
        called_query, called_top_k = index.search.call_args.args
        np.testing.assert_array_equal(called_query, query)
        self.assertEqual(called_top_k, 7)


if __name__ == "__main__":
    unittest.main()
