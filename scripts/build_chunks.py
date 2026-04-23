from __future__ import annotations

from pathlib import Path

from app.corpus_loader import load_documents
from app.chunking import FixedChunker
from app.io_utils import write_jsonl

""" 
This script is responsible for building the chunks out of the stored corpus and queries.
"""


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

INPUT_PATH_DEV = DATA_DIR / "processed/hotpot_corpus_dev.jsonl"
OUTPUT_PATH_DEV = DATA_DIR / "built/chunks_dev.jsonl"

INPUT_PATH_COMPLETE = DATA_DIR / "processed/hotpot_corpus.jsonl"
OUTPUT_PATH_COMPLETE = DATA_DIR / "built/chunks.jsonl"

# Boolean that decides whether the chunks are built for the development subset or the complete dataset
build_dev = False

# Temporary implementation choice:
# Using the Qwen tokenizer for chunk boundary calculation.
# TODO: verify if I want to keep this choice
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"

def main():
    if build_dev:
        input_path = INPUT_PATH_DEV
        output_path = OUTPUT_PATH_DEV
    else:
        input_path = INPUT_PATH_COMPLETE
        output_path = OUTPUT_PATH_COMPLETE


    print(f"Loading documents from {str(input_path)} ...")
    docs = load_documents(str(input_path))
    print(f"Loaded {len(docs)} documents.")

    chunker = FixedChunker(
        tokenizer_name=TOKENIZER_NAME,
        chunk_size=256,
        overlap=32,
    )

    print("Chunking documents...")
    chunks = chunker.chunk_documents(docs)

    print(f"Built {len(chunks)} chunks.")
    write_jsonl(str(output_path), (chunk.model_dump() for chunk in chunks))
    print(f"Saved chunks to {str(output_path)}")


if __name__ == "__main__":
    main()