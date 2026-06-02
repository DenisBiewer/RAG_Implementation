from __future__ import annotations

"""
Build fixed-size chunks from a processed corpus JSONL file.

Presets follow the project data layout:
- clean datasets: data/processed/clean/<dataset>
- attacked datasets: data/processed/attacks/<attack>/<dataset>

Use explicit `--input-path` and `--output-path` to override presets.
"""

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.indexing.indexing_presets import (
    INDEXING_PRESETS,
    TOKENIZER_NAME,
    is_attack_preset,
    resolve_chunk_paths as resolve_preset_chunk_paths,
)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve input/output paths from a preset plus optional overrides."""
    return resolve_preset_chunk_paths(
        args.preset,
        input_path=args.input_path,
        chunks_path=args.output_path,
    )


def build_chunks(
    *,
    input_path: Path,
    output_path: Path,
    tokenizer_name: str,
    chunk_size: int,
    overlap: int,
) -> None:
    """Build token chunks from one processed corpus file."""
    from app.chunking import FixedChunker
    from app.corpus_loader import load_documents
    from app.io_utils import write_jsonl

    print(f"Loading documents from {input_path}...")
    docs = load_documents(str(input_path))
    print(f"Loaded {len(docs)} documents.")

    chunker = FixedChunker(
        tokenizer_name=tokenizer_name,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    print("Chunking documents...")
    chunks = chunker.chunk_documents(docs)

    print(f"Built {len(chunks)} chunks.")
    write_jsonl(str(output_path), (chunk.model_dump() for chunk in chunks))
    print(f"Saved chunks to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Build token chunks from a processed corpus.",
        epilog=(
            "Warning: `--tokenizer-name`, `--chunk-size`, `--overlap`, and the "
            "effective input corpus/order are retrieval-artifact compatibility "
            "settings. Reuse the same values when building attack artifacts from "
            "an existing clean retrieval base."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted(INDEXING_PRESETS),
        default="clean_complete",
        help="Path preset to use. Defaults to the canonical complete clean corpus.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional corpus JSONL input path overriding the preset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional chunks JSONL output path overriding the preset.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=TOKENIZER_NAME,
        help="Tokenizer used to compute chunk boundaries.",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if is_attack_preset(args.preset):
        raise ValueError(
            "Attack presets must be built through "
            "`scripts/indexing/build_retrieval_artifacts.py` so the attacked "
            "chunks remain an append-only derivative of the clean retrieval base."
        )
    input_path, output_path = resolve_paths(args)
    build_chunks(
        input_path=input_path,
        output_path=output_path,
        tokenizer_name=args.tokenizer_name,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
