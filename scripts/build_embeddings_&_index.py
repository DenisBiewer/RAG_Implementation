from __future__ import annotations

from pathlib import Path

from app.corpus_loader import load_chunks
from app.embedder import ContrieverEmbedder
from app.vector_index import FaissIndex

BASE_DIR = Path(__file__).resolve().parent
BUILT_DIR = BASE_DIR.parent / "data" / "built"

INPUT_PATH_CHUNKS_DEV = BUILT_DIR / "chunks_dev.jsonl"
INDEX_PATH_DEV = BUILT_DIR / "faiss_dev.index"
META_PATH_DEV = BUILT_DIR / "faiss_dev_meta.json"

INPUT_PATH_CHUNKS_COMPLETE = BUILT_DIR / "chunks.jsonl"
INDEX_PATH_COMPLETE = BUILT_DIR / "faiss.index"
META_PATH_COMPLETE = BUILT_DIR / "faiss_meta.json"

DEVICE = "cpu"

# Boolean that decides whether the chunks are built for the development subset or the complete dataset
build_dev = False

def main():
    if build_dev:
        input_path_chunks = INPUT_PATH_CHUNKS_DEV
        index_path = INDEX_PATH_DEV
        meta_path = META_PATH_DEV
    else:
        input_path_chunks = INPUT_PATH_CHUNKS_COMPLETE
        index_path = INDEX_PATH_COMPLETE
        meta_path = META_PATH_COMPLETE

    print(f"Loading chunks from {input_path_chunks} ...")
    chunks = load_chunks(str(input_path_chunks))
    print(f"Loaded {len(chunks)} chunks.")

    print("Loading Contriever embedder...")
    embedder = ContrieverEmbedder(device=DEVICE)

    print("Encoding chunk texts...")
    embeddings = embedder.encode([chunk.text for chunk in chunks], batch_size=16)

    print("Embedding shape:", embeddings.shape)

    index = FaissIndex(dim=embeddings.shape[1])
    print("Adding embeddings to FAISS...")
    index.add(embeddings, chunks)

    print("Saving index...")
    index.save(str(index_path), str(meta_path))

    print(f"Saved FAISS index to {str(index_path)}")
    print(f"Saved metadata mapping to {str(meta_path)}")


if __name__ == "__main__":
    main()