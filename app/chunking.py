from __future__ import annotations

from transformers import AutoTokenizer

from app.schemas import Document, Chunk


class FixedChunker:
    """
    Fixed-size chunker with overlap
    """

    def __init__(self, tokenizer_name: str, chunk_size: int = 256, overlap: int = 32):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """
            Convert one document into multiple token-based chunks.
        """
        encoded = self.tokenizer(
            doc.text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        input_ids = encoded["input_ids"]

        if not input_ids:
            return []

        chunks: list[Chunk] = []
        start = 0
        chunk_idx = 0

        while start < len(input_ids):
            end = min(start + self.chunk_size, len(input_ids))
            chunk_ids = input_ids[start:end]

            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}::chunk_{chunk_idx}",
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        token_start=start,
                        token_end=end,
                        document_type="wiki_chunk" if doc.document_type == "wiki_article" else doc.document_type,
                        topic_id=doc.topic_id,
                        timestamp=doc.timestamp,
                        is_malicious=doc.is_malicious,
                        attack_type=doc.attack_type,
                    )
                )

            if end == len(input_ids):
                break

            start = end - self.overlap
            chunk_idx += 1

        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks