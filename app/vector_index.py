from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.schemas import Chunk


class FaissIndex:
    """
    Exact FAISS index backed by inner-product similarity.

    This wrapper assumes embeddings are already L2-normalized. Under that
    assumption, `faiss.IndexFlatIP` performs exact cosine-similarity search.
    Metadata is stored separately in `row_to_chunk`, keyed by the FAISS row id.
    """

    def __init__(self, dim: int):
        """
        Create an empty index for vectors of a fixed dimensionality.

        Args:
            dim: Embedding dimension expected by the index.
        """
        self.index = faiss.IndexFlatIP(dim)
        self.row_to_chunk: dict[int, dict[str, Any]] = {}

    def _as_faiss_matrix(
        self,
        embeddings: np.ndarray,
        *,
        name: str,
        allow_single_vector: bool = False,
    ) -> np.ndarray:
        """
        Validate and convert input vectors to the contiguous float32 matrix FAISS expects.
        """
        matrix = np.asarray(embeddings, dtype=np.float32)

        if allow_single_vector and matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        if matrix.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D array with shape (n_vectors, {self.index.d})."
            )

        if matrix.shape[1] != self.index.d:
            raise ValueError(
                f"{name} must have dimension {self.index.d}, got {matrix.shape[1]}."
            )

        return np.ascontiguousarray(matrix)

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """
        Append chunk embeddings and their metadata to the index.

        Args:
            embeddings: Matrix of shape `(n_chunks, dim)` in float-compatible format.
            chunks: Chunk metadata aligned row-for-row with `embeddings`.

        Raises:
            ValueError: If the number of embeddings does not match `chunks`, or if
                the embedding matrix has the wrong shape.
        """
        embeddings = self._as_faiss_matrix(embeddings, name="embeddings")

        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")

        if not chunks:
            return

        start_row = len(self.row_to_chunk)
        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
            self.row_to_chunk[start_row + i] = chunk.model_dump()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Search the index with a single query embedding.

        Args:
            query_embedding: One query vector of shape `(dim,)` or `(1, dim)`.
            top_k: Maximum number of matches to return.

        Returns:
            A pair `(scores, indices)` for the single query. If the index is empty,
            both arrays are empty. `top_k` is capped to the number of indexed rows so
            the caller never receives FAISS sentinel indices such as `-1`.

        Raises:
            ValueError: If `top_k` is not positive, or if the query shape is invalid.
        """
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if self.index.ntotal == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

        query = self._as_faiss_matrix(
            query_embedding,
            name="query_embedding",
            allow_single_vector=True,
        )

        if query.shape[0] != 1:
            raise ValueError("query_embedding must contain exactly one query vector.")

        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
        return scores[0], indices[0]

    def reconstruct(self, row_id: int) -> np.ndarray:
        """
        Reconstruct one stored embedding vector from the FAISS index.

        Args:
            row_id: Row identifier of the stored vector.

        Returns:
            The reconstructed embedding as a contiguous `float32` NumPy array.

        Raises:
            KeyError: If `row_id` is not present in the index metadata.

        Notes:
            MMR needs candidate-to-candidate similarity values. Reconstructing
            vectors from the FAISS index keeps that logic local to the wrapper
            instead of requiring callers to reach into the raw FAISS object.
        """
        if row_id not in self.row_to_chunk:
            raise KeyError(f"Unknown FAISS row id: {row_id}")

        return np.asarray(self.index.reconstruct(row_id), dtype=np.float32)

    def save(self, index_path: str | Path, meta_path: str | Path) -> None:
        """
        Persist the FAISS index and the row-to-chunk metadata mapping to disk.
        """
        index_path = Path(index_path)
        meta_path = Path(meta_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.row_to_chunk, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_path: str | Path, meta_path: str | Path) -> "FaissIndex":
        """
        Reconstruct a populated index from files previously written by `save`.

        This is a class method because loading must create and return a new
        instance from serialized state; using `cls` also keeps subclass loading
        correct without hard-coding `FaissIndex`.
        """
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        index = faiss.read_index(str(index_path))

        with meta_path.open("r", encoding="utf-8") as f:
            row_to_chunk = json.load(f)

        row_to_chunk = {int(k): v for k, v in row_to_chunk.items()}

        if set(row_to_chunk) != set(range(index.ntotal)):
            raise ValueError(
                "Loaded metadata rows do not match the FAISS index rows."
            )

        obj = cls(index.d)
        obj.index = index
        obj.row_to_chunk = row_to_chunk
        return obj
