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


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.indexing.indexing_presets import (
    INDEXING_PRESETS,
    is_attack_preset,
    resolve_index_paths as resolve_preset_index_paths,
)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve chunks/index/meta paths from a preset plus optional overrides."""
    return resolve_preset_index_paths(
        args.preset,
        chunks_path=args.chunks_path,
        index_path=args.index_path,
        meta_path=args.meta_path,
    )


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
    parser = argparse.ArgumentParser(
        description="Build embeddings and a FAISS index from chunks.",
        epilog=(
            "Note: `--device` and `--batch-size` are performance settings. They "
            "should not change the logical corpus structure, but CPU and CUDA runs "
            "may not guarantee bit-for-bit identical embeddings."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted(INDEXING_PRESETS),
        default="clean_complete",
        help="Path preset to use. Defaults to the canonical complete clean chunks.",
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
    if is_attack_preset(args.preset):
        raise ValueError(
            "Attack presets must be built through "
            "`scripts/indexing/build_retrieval_artifacts.py` so the attacked "
            "index remains an append-only derivative of the clean retrieval base."
        )
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
