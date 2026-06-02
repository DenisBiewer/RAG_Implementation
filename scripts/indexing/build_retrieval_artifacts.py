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


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.indexing.build_chunks import build_chunks
from scripts.indexing.build_embeddings_index import build_embeddings_index, resolve_device
from scripts.indexing.attack_artifact_utils import (
    PERFORMANCE_SETTINGS_NOTE,
    RETRIEVAL_ARTIFACT_COMPATIBILITY_WARNING,
    ensure_attack_output_paths_distinct,
    extract_appended_records,
    load_build_info,
    validate_base_build_settings,
    write_appended_jsonl,
    write_build_info,
)
from scripts.indexing.indexing_presets import (
    INDEXING_PRESETS,
    TOKENIZER_NAME,
    is_attack_preset,
    resolve_attack_base_paths,
    resolve_retrieval_artifact_paths,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Build chunks and FAISS retrieval artifacts for one preset.",
        epilog=(
            "Warning: `--tokenizer-name`, `--chunk-size`, `--overlap`, and the "
            "effective input corpus/order are retrieval-artifact compatibility "
            "settings. If you changed any of these when building the clean "
            "dataset, you must use the same values when building attack artifacts. "
            "Also note that `--device` and `--batch-size` are performance settings "
            "rather than structural compatibility settings, though CPU and CUDA "
            "runs may not guarantee bit-for-bit identical embeddings."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted(INDEXING_PRESETS),
        default="clean_complete",
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
    return resolve_retrieval_artifact_paths(
        args.preset,
        input_path=args.input_path,
        chunks_path=args.chunks_path,
        index_path=args.index_path,
        meta_path=args.meta_path,
    )


def build_attack_retrieval_artifacts(
    *,
    preset_name: str,
    attack_input_path: Path,
    chunks_path: Path,
    index_path: Path,
    meta_path: Path,
    tokenizer_name: str,
    chunk_size: int,
    overlap: int,
    device: str,
    batch_size: int,
) -> None:
    """Build attack retrieval artifacts as an append-only derivative of a clean base."""
    from app.chunking import FixedChunker
    from app.embedder import ContrieverEmbedder
    from app.schemas import Document
    from app.vector_index import FaissIndex

    (
        base_preset_name,
        base_input_path,
        base_chunks_path,
        base_index_path,
        base_meta_path,
    ) = resolve_attack_base_paths(preset_name)

    print(f"Building attack retrieval artifacts for preset '{preset_name}'")
    print(f"- clean retrieval base preset: {base_preset_name}")
    print(f"- clean base corpus: {base_input_path}")
    print(f"- attacked corpus: {attack_input_path}")
    print(RETRIEVAL_ARTIFACT_COMPATIBILITY_WARNING)
    print(PERFORMANCE_SETTINGS_NOTE)

    ensure_attack_output_paths_distinct(
        base_chunks_path=base_chunks_path,
        base_index_path=base_index_path,
        base_meta_path=base_meta_path,
        chunks_path=chunks_path,
        index_path=index_path,
        meta_path=meta_path,
    )

    base_build_info = load_build_info(base_meta_path)
    build_settings_warning = validate_base_build_settings(
        base_build_info,
        tokenizer_name=tokenizer_name,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if build_settings_warning:
        print(f"Warning: {build_settings_warning}")

    clean_doc_count, appended_records = extract_appended_records(
        base_path=base_input_path,
        attacked_path=attack_input_path,
    )
    attack_docs = [Document(**record) for record in appended_records]
    malicious_doc_count = sum(1 for record in appended_records if record.get("is_malicious"))

    print(
        "Validated attacked corpus as clean-prefix plus appended suffix: "
        f"{clean_doc_count} clean docs + {len(attack_docs)} appended docs."
    )

    chunker = FixedChunker(
        tokenizer_name=tokenizer_name,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    print("Chunking appended attack documents...")
    appended_chunks = chunker.chunk_documents(attack_docs)
    print(f"Built {len(appended_chunks)} appended chunks.")

    write_appended_jsonl(
        prefix_path=base_chunks_path,
        output_path=chunks_path,
        suffix_records=(chunk.model_dump() for chunk in appended_chunks),
    )
    print(f"Saved attacked chunks to {chunks_path}")

    print("Loading clean retrieval base index...")
    index = FaissIndex.load(str(base_index_path), str(base_meta_path))
    if appended_chunks:
        print(f"Loading Contriever embedder on {device}...")
        embedder = ContrieverEmbedder(device=device)

        print("Encoding appended chunk texts...")
        embeddings = embedder.encode(
            [chunk.text for chunk in appended_chunks],
            batch_size=batch_size,
        )
        print("Appended embedding shape:", embeddings.shape)

        print("Appending embeddings to FAISS...")
        index.add(embeddings, appended_chunks)

    print("Saving attacked index...")
    index.save(str(index_path), str(meta_path))

    write_build_info(
        meta_path,
        {
            "preset": preset_name,
            "build_mode": "attack_append",
            "input_path": str(attack_input_path),
            "chunks_path": str(chunks_path),
            "index_path": str(index_path),
            "meta_path": str(meta_path),
            "tokenizer_name": tokenizer_name,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "device": device,
            "batch_size": batch_size,
            "base_preset": base_preset_name,
            "base_input_path": str(base_input_path),
            "base_chunks_path": str(base_chunks_path),
            "base_index_path": str(base_index_path),
            "base_meta_path": str(base_meta_path),
            "clean_prefix_document_count": clean_doc_count,
            "appended_document_count": len(attack_docs),
            "appended_malicious_document_count": malicious_doc_count,
            "appended_chunk_count": len(appended_chunks),
        },
    )
    print(f"Saved attacked FAISS index to {index_path}")
    print(f"Saved attacked metadata mapping to {meta_path}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    input_path, chunks_path, index_path, meta_path = resolve_paths(args)
    device = resolve_device(args.device)

    if is_attack_preset(args.preset):
        build_attack_retrieval_artifacts(
            preset_name=args.preset,
            attack_input_path=input_path,
            chunks_path=chunks_path,
            index_path=index_path,
            meta_path=meta_path,
            tokenizer_name=args.tokenizer_name,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            device=device,
            batch_size=args.batch_size,
        )
    else:
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
        write_build_info(
            meta_path,
            {
                "preset": args.preset,
                "build_mode": "full",
                "input_path": str(input_path),
                "chunks_path": str(chunks_path),
                "index_path": str(index_path),
                "meta_path": str(meta_path),
                "tokenizer_name": args.tokenizer_name,
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "device": device,
                "batch_size": args.batch_size,
            },
        )
    print("Finished building retrieval artifacts.")


if __name__ == "__main__":
    main()
