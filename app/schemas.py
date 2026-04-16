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