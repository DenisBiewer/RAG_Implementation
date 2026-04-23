# app/corpus_loader.py
from __future__ import annotations

from app.io_utils import read_jsonl
from app.schemas import Document, QueryRecord, Chunk

def load_documents(path: str) -> list[Document]:
    return [Document(**record) for record in read_jsonl(path)]

def load_queries(path: str) -> list[QueryRecord]:
    return [QueryRecord(**record) for record in read_jsonl(path)]

def load_chunks(path: str) -> list[Chunk]:
    return [Chunk(**record) for record in read_jsonl(path)]