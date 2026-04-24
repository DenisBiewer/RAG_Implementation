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


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.corpus_loader import load_documents
from app.chunking import FixedChunker
from app.io_utils import write_jsonl


DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
BUILT_DIR = DATA_DIR / "built"

TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"

PRESETS = {
    "clean_dev": {
        "input_path": PROCESSED_DIR / "clean" / "dev" / "corpus.jsonl",
        "output_path": BUILT_DIR / "clean" / "dev" / "chunks.jsonl",
    },
    "clean_controlled_dev": {
        "input_path": PROCESSED_DIR / "clean" / "controlled_dev" / "corpus.jsonl",
        "output_path": BUILT_DIR / "clean" / "controlled_dev" / "chunks.jsonl",
    },
    "clean_complete": {
        "input_path": PROCESSED_DIR / "clean" / "complete" / "corpus.jsonl",
        "output_path": BUILT_DIR / "clean" / "complete" / "chunks.jsonl",
    },
    "instruction_following_controlled_dev": {
        "input_path": PROCESSED_DIR / "attacks" / "instruction_following" / "controlled_dev" / "corpus.jsonl",
        "output_path": BUILT_DIR / "attacks" / "instruction_following" / "controlled_dev" / "chunks.jsonl",
    },
}


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve input/output paths from a preset plus optional overrides."""
    preset = PRESETS[args.preset]
    input_path = args.input_path or preset["input_path"]
    output_path = args.output_path or preset["output_path"]
    return input_path, output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Build token chunks from a processed corpus.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="clean_controlled_dev",
        help="Path preset to use. Defaults to the clean controlled development corpus.",
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
    input_path, output_path = resolve_paths(args)

    print(f"Loading documents from {input_path}...")
    docs = load_documents(str(input_path))
    print(f"Loaded {len(docs)} documents.")

    chunker = FixedChunker(
        tokenizer_name=args.tokenizer_name,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print("Chunking documents...")
    chunks = chunker.chunk_documents(docs)

    print(f"Built {len(chunks)} chunks.")
    write_jsonl(str(output_path), (chunk.model_dump() for chunk in chunks))
    print(f"Saved chunks to {output_path}")


if __name__ == "__main__":
    main()
