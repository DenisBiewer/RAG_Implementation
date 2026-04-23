from __future__ import annotations

"""
Retrieval utilities for the current RAG pipeline.

This module contains:
- the plain baseline dense top-k retrieval path
- retrieval-stage defense helpers for MMR and rule-based suspicion filtering

The defended retrieval path is intentionally transparent and auditable. It
returns both the final retrieved chunks and a structured dictionary describing
which defense steps were applied and how they affected the ranking.
"""

from dataclasses import dataclass

import numpy as np

from app.schemas import DefenseConfig, RetrievedChunk
from app.suspicion import score_text_suspicion


MMR_CANDIDATE_MULTIPLIER = 4


@dataclass(frozen=True)
class RetrievalCandidate:
    """
    Internal representation of a retrieved candidate before final ranking.

    Attributes:
        row_id: FAISS row id of the candidate.
        chunk: Structured chunk metadata for downstream prompt construction.
        embedding: Reconstructed dense embedding for similarity-based reranking.
        original_score: Raw similarity score returned by the FAISS search.
    """

    row_id: int
    chunk: RetrievedChunk
    embedding: np.ndarray
    original_score: float


def _candidate_pool_size(top_k: int) -> int:
    """
    Compute the initial retrieval pool size used before defense reranking.

    MMR only has meaningful room to diversify results if it can choose from a
    pool larger than the final output size. The final output remains fixed at
    `top_k`, but the initial dense retrieval pool is expanded.
    """
    return max(top_k, top_k * MMR_CANDIDATE_MULTIPLIER)


def _build_retrieved_chunk(meta: dict, score: float, rank: int) -> RetrievedChunk:
    """Create a `RetrievedChunk` object from FAISS metadata and a retrieval score."""
    return RetrievedChunk(
        chunk_id=meta["chunk_id"],
        doc_id=meta["doc_id"],
        text=meta["text"],
        score=float(score),
        rank=rank,
        is_malicious=meta["is_malicious"],
        attack_type=meta.get("attack_type"),
        document_type=meta["document_type"],
        topic_id=meta["topic_id"],
    )


def retrieve_top_k(index, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievedChunk]:
    """
    Baseline dense retrieval without any defenses.

    Args:
        index: Loaded `FaissIndex`.
        query_embedding: Single normalized query embedding.
        top_k: Number of results to return.

    Returns:
        The top-k highest-scoring chunks from exact dense retrieval.
    """
    scores, indices = index.search(query_embedding, top_k)

    results: list[RetrievedChunk] = []

    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        if idx < 0:
            continue

        meta = index.row_to_chunk[int(idx)]
        results.append(_build_retrieved_chunk(meta, float(score), rank))

    return results


def _retrieve_candidates(index, query_embedding: np.ndarray, top_k: int) -> list[RetrievalCandidate]:
    """
    Retrieve an oversampled dense candidate pool for defense-stage reranking.

    Args:
        index: Loaded `FaissIndex`.
        query_embedding: Single normalized query embedding.
        top_k: Final number of results the caller intends to keep.

    Returns:
        A list of candidates sorted by raw dense similarity.
    """
    scores, indices = index.search(query_embedding, _candidate_pool_size(top_k))
    candidates: list[RetrievalCandidate] = []

    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        if idx < 0:
            continue

        row_id = int(idx)
        meta = index.row_to_chunk[row_id]
        candidates.append(
            RetrievalCandidate(
                row_id=row_id,
                chunk=_build_retrieved_chunk(meta, float(score), rank),
                embedding=index.reconstruct(row_id),
                original_score=float(score),
            )
        )

    return candidates


def _apply_security_filtering(
    candidates: list[RetrievalCandidate],
    suspicion_threshold: float,
) -> tuple[list[tuple[float, RetrievalCandidate]], dict]:
    """
    Down-rank suspicious candidates using rule-based local heuristics.

    Candidates at or above the configured suspicion threshold are not removed.
    Instead, their dense similarity scores are discounted to preserve a softer,
    more auditable defense that still allows evidence to survive if needed.
    """
    reranked: list[tuple[float, RetrievalCandidate]] = []
    scored_candidates: list[dict] = []

    for candidate in candidates:
        suspicion = score_text_suspicion(candidate.chunk.text)
        adjusted_score = candidate.original_score

        if suspicion.score >= suspicion_threshold:
            # Discount highly suspicious chunks instead of deleting them.
            adjusted_score = candidate.original_score * (1.0 - suspicion.score)

        reranked.append((adjusted_score, candidate))
        scored_candidates.append(
            {
                "chunk_id": candidate.chunk.chunk_id,
                "topic_id": candidate.chunk.topic_id,
                "original_score": round(candidate.original_score, 6),
                "adjusted_score": round(adjusted_score, 6),
                "suspicion_score": round(suspicion.score, 6),
                "matched_rules": suspicion.matched_rules,
                "downranked": suspicion.score >= suspicion_threshold,
            }
        )

    reranked.sort(key=lambda item: item[0], reverse=True)

    return reranked, {
        "suspicion_threshold": suspicion_threshold,
        "candidates": scored_candidates,
    }


def _mmr_select(
    scored_candidates: list[tuple[float, RetrievalCandidate]],
    top_k: int,
    lambda_param: float,
) -> tuple[list[RetrievalCandidate], dict]:
    """
    Select a diverse final subset with Maximum Marginal Relevance.

    Args:
        scored_candidates: Candidate list paired with the relevance score to use
            for MMR. This may be the raw dense score or a security-adjusted one.
        top_k: Final number of chunks to return.
        lambda_param: Relevance-vs-diversity tradeoff. Lower values emphasize
            diversity more strongly.

    Returns:
        A pair `(selected_candidates, decision_log)`.
    """
    if not scored_candidates:
        return [], {"lambda": lambda_param, "selection_trace": []}

    selected: list[tuple[float, RetrievalCandidate]] = []
    remaining = list(scored_candidates)
    selection_trace: list[dict] = []

    while remaining and len(selected) < top_k:
        best_position = 0
        best_mmr_score = float("-inf")

        for position, (relevance_score, candidate) in enumerate(remaining):
            if not selected:
                redundancy_penalty = 0.0
            else:
                redundancy_penalty = max(
                    float(np.dot(candidate.embedding, chosen.embedding))
                    for _, chosen in selected
                )

            mmr_score = (lambda_param * relevance_score) - (
                (1.0 - lambda_param) * redundancy_penalty
            )

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_position = position

        chosen_score, chosen_candidate = remaining.pop(best_position)
        selected.append((chosen_score, chosen_candidate))
        selection_trace.append(
            {
                "chunk_id": chosen_candidate.chunk.chunk_id,
                "topic_id": chosen_candidate.chunk.topic_id,
                "relevance_score": round(chosen_score, 6),
                "mmr_score": round(best_mmr_score, 6),
                "selection_rank": len(selected),
            }
        )

    return [candidate for _, candidate in selected], {
        "lambda": lambda_param,
        "selection_trace": selection_trace,
    }


def _rank_selected_chunks(
    selected_candidates: list[RetrievalCandidate],
    selected_scores: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
    """
    Convert selected internal candidates back into final `RetrievedChunk` objects.

    Args:
        selected_candidates: Candidates chosen for the final result set.
        selected_scores: Optional mapping from `chunk_id` to the final score that
            should be exposed in the returned `RetrievedChunk`.
    """
    results: list[RetrievedChunk] = []

    for final_rank, candidate in enumerate(selected_candidates, start=1):
        exposed_score = candidate.original_score
        if selected_scores is not None:
            exposed_score = selected_scores.get(candidate.chunk.chunk_id, exposed_score)

        results.append(
            candidate.chunk.model_copy(
                update={
                    "rank": final_rank,
                    "score": float(exposed_score),
                }
            )
        )

    return results


def retrieve_with_defenses(
    index,
    query_embedding: np.ndarray,
    top_k: int,
    defense: DefenseConfig,
    *,
    mmr_lambda: float | None = None,
    suspicion_threshold: float | None = None,
) -> tuple[list[RetrievedChunk], dict]:
    """
    Run retrieval with optional MMR and security-aware down-ranking.

    Args:
        index: Loaded `FaissIndex`.
        query_embedding: Single normalized query embedding.
        top_k: Final number of retrieved chunks to return.
        defense: Defense toggle configuration.
        mmr_lambda: Lambda parameter for MMR if enabled.
        suspicion_threshold: Threshold above which suspicious chunks are
            down-ranked if retrieval filtering is enabled.

    Returns:
        A pair `(retrieved_chunks, defense_decisions)` where the first element is
        the final result set and the second contains audit information.
    """
    candidates = _retrieve_candidates(index, query_embedding, top_k)
    if not candidates:
        return [], {"defenses_enabled": []}

    defense_decisions: dict = {"defenses_enabled": []}
    scored_candidates = [(candidate.original_score, candidate) for candidate in candidates]

    if defense.use_retrieval_filtering:
        if suspicion_threshold is None:
            raise ValueError("suspicion_threshold must be provided when retrieval filtering is enabled.")

        scored_candidates, filtering_log = _apply_security_filtering(candidates, suspicion_threshold)
        defense_decisions["defenses_enabled"].append("retrieval_filtering")
        defense_decisions["retrieval_filtering"] = filtering_log

    if defense.use_mmr:
        if mmr_lambda is None:
            raise ValueError("mmr_lambda must be provided when MMR is enabled.")

        selected_candidates, mmr_log = _mmr_select(scored_candidates, top_k, mmr_lambda)
        final_score_map = {
            candidate.chunk.chunk_id: float(score)
            for score, candidate in scored_candidates
        }
        defense_decisions["defenses_enabled"].append("mmr")
        defense_decisions["mmr"] = mmr_log
        return _rank_selected_chunks(selected_candidates, final_score_map), defense_decisions

    scored_candidates = scored_candidates[:top_k]
    final_candidates = [candidate for _, candidate in scored_candidates]
    final_score_map = {
        candidate.chunk.chunk_id: float(score)
        for score, candidate in scored_candidates
    }
    return _rank_selected_chunks(final_candidates, final_score_map), defense_decisions
