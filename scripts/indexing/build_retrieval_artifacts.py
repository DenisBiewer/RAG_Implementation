from __future__ import annotations

"""
Build both retrieval artifacts for one dataset preset:
- token chunks
- embedding matrix plus FAISS index

This is the normal entry point for retrieval preparation.
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

from build_chunks import PRESETS as CHUNK_PRESETS
from build_chunks import TOKENIZER_NAME, build_chunks
from build_embeddings_index import build_embeddings_index, resolve_device


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Build chunks and FAISS retrieval artifacts for one preset."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(CHUNK_PRESETS),
        default="clean_controlled_dev",
        help="Dataset preset to build retrieval artifacts for.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional corpus JSONL input path overriding the preset.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=None,
        help="Optional chunks JSONL output path overriding the preset.",
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
        "--tokenizer-name",
        default=TOKENIZER_NAME,
        help="Tokenizer used to compute chunk boundaries.",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument(
        "--device",
        default="auto",
        help='Embedding device: "auto", "cpu", or "cuda".',
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve corpus, chunks, index, and metadata paths."""
    chunk_preset = CHUNK_PRESETS[args.preset]
    input_path = args.input_path or chunk_preset["input_path"]
    chunks_path = args.chunks_path or chunk_preset["output_path"]
    index_path = args.index_path or chunks_path.with_name("faiss.index")
    meta_path = args.meta_path or chunks_path.with_name("faiss_meta.json")
    return input_path, chunks_path, index_path, meta_path


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    input_path, chunks_path, index_path, meta_path = resolve_paths(args)
    device = resolve_device(args.device)

    print(f"Building retrieval artifacts for preset '{args.preset}'")
    build_chunks(
        input_path=input_path,
        output_path=chunks_path,
        tokenizer_name=args.tokenizer_name,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    build_embeddings_index(
        chunks_path=chunks_path,
        index_path=index_path,
        meta_path=meta_path,
        device=device,
        batch_size=args.batch_size,
    )
    print("Finished building retrieval artifacts.")


if __name__ == "__main__":
    main()
