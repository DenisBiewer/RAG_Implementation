from __future__ import annotations

"""
Build a Contriever embedding matrix and FAISS index from chunk JSONL.

Presets follow the project data layout:
- clean datasets: data/built/clean/<dataset>
- attacked datasets: data/built/attacks/<attack>/<dataset>

Use explicit paths to override presets when needed.
"""

import argparse
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = next(
    parent
    for parent in (BASE_DIR, *BASE_DIR.parents)
    if (parent / "app").exists() and (parent / "data").exists()
)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


BUILT_DIR = PROJECT_DIR / "data" / "built"

PRESETS = {
    "clean_dev": {
        "chunks_path": BUILT_DIR / "clean" / "dev" / "chunks.jsonl",
        "index_path": BUILT_DIR / "clean" / "dev" / "faiss.index",
        "meta_path": BUILT_DIR / "clean" / "dev" / "faiss_meta.json",
    },
    "close_ended_dev": {
        "chunks_path": BUILT_DIR / "clean" / "close_ended_dev" / "initial" / "chunks.jsonl",
        "index_path": BUILT_DIR / "clean" / "close_ended_dev" / "initial" / "faiss.index",
        "meta_path": BUILT_DIR / "clean" / "close_ended_dev" / "initial" / "faiss_meta.json",
    },
    "close_ended_dev_final": {
        "chunks_path": BUILT_DIR / "clean" / "close_ended_dev" / "final" / "chunks.jsonl",
        "index_path": BUILT_DIR / "clean" / "close_ended_dev" / "final" / "faiss.index",
        "meta_path": BUILT_DIR / "clean" / "close_ended_dev" / "final" / "faiss_meta.json",
    },
    "clean_controlled_dev": {
        "chunks_path": BUILT_DIR / "clean" / "controlled_dev" / "chunks.jsonl",
        "index_path": BUILT_DIR / "clean" / "controlled_dev" / "faiss.index",
        "meta_path": BUILT_DIR / "clean" / "controlled_dev" / "faiss_meta.json",
    },
    "clean_complete": {
        "chunks_path": BUILT_DIR / "clean" / "complete" / "chunks.jsonl",
        "index_path": BUILT_DIR / "clean" / "complete" / "faiss.index",
        "meta_path": BUILT_DIR / "clean" / "complete" / "faiss_meta.json",
    },
    "instruction_following_controlled_dev": {
        "chunks_path": BUILT_DIR / "attacks" / "instruction_following" / "controlled_dev" / "chunks.jsonl",
        "index_path": BUILT_DIR / "attacks" / "instruction_following" / "controlled_dev" / "faiss.index",
        "meta_path": BUILT_DIR / "attacks" / "instruction_following" / "controlled_dev" / "faiss_meta.json",
    },
    "exfiltration_controlled_dev": {
        "chunks_path": BUILT_DIR / "attacks" / "exfiltration" / "controlled_dev" / "chunks.jsonl",
        "index_path": BUILT_DIR / "attacks" / "exfiltration" / "controlled_dev" / "faiss.index",
        "meta_path": BUILT_DIR / "attacks" / "exfiltration" / "controlled_dev" / "faiss_meta.json",
    },
    "exfiltration_controlled_dev_control": {
        "chunks_path": BUILT_DIR / "attacks" / "exfiltration" / "controlled_dev_control" / "chunks.jsonl",
        "index_path": BUILT_DIR / "attacks" / "exfiltration" / "controlled_dev_control" / "faiss.index",
        "meta_path": BUILT_DIR / "attacks" / "exfiltration" / "controlled_dev_control" / "faiss_meta.json",
    },
    "poisoned_rag_close_ended_dev": {
        "chunks_path": BUILT_DIR / "attacks" / "poisoned_rag" / "close_ended_dev" / "chunks.jsonl",
        "index_path": BUILT_DIR / "attacks" / "poisoned_rag" / "close_ended_dev" / "faiss.index",
        "meta_path": BUILT_DIR / "attacks" / "poisoned_rag" / "close_ended_dev" / "faiss_meta.json",
    },
}


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve chunks/index/meta paths from a preset plus optional overrides."""
    preset = PRESETS[args.preset]
    chunks_path = args.chunks_path or preset["chunks_path"]
    index_path = args.index_path or preset["index_path"]
    meta_path = args.meta_path or preset["meta_path"]
    return chunks_path, index_path, meta_path


def build_embeddings_index(
    *,
    chunks_path: Path,
    index_path: Path,
    meta_path: Path,
    device: str,
    batch_size: int,
) -> None:
    """Build embeddings and a FAISS index from one chunk file."""
    from app.corpus_loader import load_chunks
    from app.embedder import ContrieverEmbedder
    from app.vector_index import FaissIndex

    print(f"Loading chunks from {chunks_path}...")
    chunks = load_chunks(str(chunks_path))
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading Contriever embedder on {device}...")
    embedder = ContrieverEmbedder(device=device)

    print("Encoding chunk texts...")
    embeddings = embedder.encode(
        [chunk.text for chunk in chunks],
        batch_size=batch_size,
    )

    print("Embedding shape:", embeddings.shape)

    index = FaissIndex(dim=embeddings.shape[1])
    print("Adding embeddings to FAISS...")
    index.add(embeddings, chunks)

    print("Saving index...")
    index.save(str(index_path), str(meta_path))

    print(f"Saved FAISS index to {index_path}")
    print(f"Saved metadata mapping to {meta_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Build embeddings and a FAISS index from chunks.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="clean_controlled_dev",
        help="Path preset to use. Defaults to the clean controlled development chunks.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=None,
        help="Optional chunks JSONL input path overriding the preset.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Optional FAISS index output path overriding the preset.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="Optional index metadata JSON output path overriding the preset.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Embedding device: "auto", "cpu", or "cuda".',
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    """Resolve the requested embedding device."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError('device must be "auto", "cpu", or "cuda"')
    return device


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    chunks_path, index_path, meta_path = resolve_paths(args)
    device = resolve_device(args.device)
    build_embeddings_index(
        chunks_path=chunks_path,
        index_path=index_path,
        meta_path=meta_path,
        device=device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
