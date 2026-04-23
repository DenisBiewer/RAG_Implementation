"""
schemas.py

Defines all core data structures used in the RAG pipeline.

These schemas ensure:
- consistent data flow across pipeline stages
- reproducibility of experiments
- structured logging for evaluation

All schemas are implemented using Pydantic for validation and clarity.
"""

from __future__ import annotations

from chunk import Chunk
from typing import Any
from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Represents a raw document in the corpus (Stage 0).

    Attributes:
        doc_id: Unique identifier of the document.
        text: Full raw text of the document.
        document_type: Type/category (e.g., policy, case, FAQ).
        topic_id: Logical grouping identifier (used for query relevance).
        timestamp: Optional creation/modification time.
        is_malicious: Whether this document is attacker-inserted (for evaluation only).
        attack_type: Type of attack associated with this document (if any).
    """
    doc_id: str
    text: str
    document_type: str
    topic_id: str
    timestamp: str | None = None
    is_malicious: bool = False
    attack_type: str | None = None

class SupportingFacts(BaseModel):
    """
    Supporting facts structure from HotpotQA

    Supporting facts are stored as a dictionary with parallel lists in the dataset:
    - title[i] is the title of a supporting article
    - sent_id[i] is the supporting sentence index in that article
    """
    title: list[str] = Field(default_factory=list)
    sent_id: list[int] = Field(default_factory=list)


class QueryRecord(BaseModel):
    """
    Evaluation query derived from HotpotQA.

    Attributes:
        query_id: Unique identifier of the query.
        question: string of the question.
        answer: Answer for the question.
        supporting_facts: SupportingFacts.
    """
    query_id: str
    question: str
    answer: str
    supporting_facts: SupportingFacts  = Field(default_factory=SupportingFacts)


class Chunk(BaseModel):
    """
    A chunked retrieval unit from a source document

    Attributes:
        chunk_id: Unique identifier of the chunk.
        doc_id: Unique identifier of the parent document.
        text: Full text of the chunk.
        token_start: starting position of the chunk.
        token_end: ending position of the chunk.
        document_type: Type/category (e.g. wikipedia article).
        topic_id: Logical grouping identifier (used for query relevance).
        timestamp: Optional creation/modification time.
        is_malicious: Whether this chunk is malicious or not (ground-truth only used for evaluation/logging).
        attack_type: type of attack the chunk is associated with if it is malicious.
    """
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


class RetrievedChunk(BaseModel):
    """
    A chunk returned by retrieval.

    `is_malicious` remains ground-truth metadata for evaluation/logging only.
    It should not be used by the baseline retrieval logic itself.

    Attributes:
        chunk_id: Unique identifier of the chunk.
        doc_id: Unique identifier of the parent document.
        text: Full text of the chunk.
        score: similarity score of the chunk (compared to the query).
        rank: position of the chunk in the sorted top-k results.
        document_type: Type/category (e.g. wikipedia article).
        topic_id: Logical grouping identifier (used for query relevance).
        timestamp: Optional creation/modification time.
        is_malicious: Whether this chunk is malicious or not (ground-truth only used for evaluation/logging).
        attack_type: type of attack the chunk is associated with if it is malicious.
    """
    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    is_malicious: bool
    attack_type: str | None = None
    document_type: str
    topic_id: str


class DefenseConfig(BaseModel):
    """
    Toggle set for the defense families described in the thesis plan.

    These flags are logged with each run even if the corresponding defense
    pipeline is not implemented yet.
    """

    use_mmr: bool = False
    use_retrieval_filtering: bool = False
    use_prompt_isolation: bool = False
    use_instruction_hierarchy: bool = False
    use_grounding: bool = False
    use_prompt_context_flags: bool = False
    use_sensitive_info_filter: bool = False
    use_policy_check: bool = False


class RunConfig(BaseModel):
    """
    Experiment configuration for one pipeline variant.

    The defense toggles are separated into `DefenseConfig` so they can be
    logged and evolved independently of the common generation parameters.
    """

    run_name: str
    seed: int = 42
    top_k: int = 5
    max_new_tokens: int = 256
    temperature: float = 0.0
    mmr_lambda: float | None = None
    suspicion_threshold: float | None = None
    defense: DefenseConfig = Field(default_factory=DefenseConfig)
