# Indexing Scripts

This directory contains the retrieval-artifact builders used by the clean and
attack workflows.

Unless noted otherwise, the `python ...` commands below assume the repo
virtualenv is already activated. If it is not activated, use
`.\.venv\Scripts\python.exe`.

## Scope

The indexing pipeline turns a corpus JSONL file into the artifacts consumed by
the runtime retriever:

- `chunks.jsonl`: chunked documents with preserved metadata
- `faiss.index`: exact FAISS inner-product index
- `faiss_meta.json`: row-to-chunk metadata aligned with the FAISS rows
- `retrieval_build_info.json`: sidecar build settings written next to
  `faiss_meta.json`

The normal entry point is `build_retrieval_artifacts.py`. The lower-level
scripts remain available when you need to inspect or rerun only one stage.

## Main Entry Points

- `build_retrieval_artifacts.py`: build both chunk and FAISS artifacts for one
  named preset
- `build_chunks.py`: chunk one corpus JSONL file into `chunks.jsonl`
- `build_embeddings_index.py`: embed chunk text and build the FAISS index plus
  metadata
- `indexing_presets.py`: source of the named clean and attack presets used by
  the builders

## Canonical Defaults

The current shared defaults are:

- chunk tokenizer: `Qwen/Qwen2.5-7B-Instruct`
- chunk size: `256`
- overlap: `32`
- embedder: `facebook/contriever`
- index type: exact `faiss.IndexFlatIP`

Because embeddings are L2-normalized in `app/embedder.py`, inner product is
cosine-equivalent for retrieval.

## Clean vs Attack Builds

Clean presets build a full retrieval artifact set from the selected corpus.

Typical complete clean build:

```powershell
python scripts/indexing/build_retrieval_artifacts.py --preset clean_complete
```

Reduced clean dataset examples:

```powershell
python scripts/indexing/build_retrieval_artifacts.py --preset close_ended_dev_initial
python scripts/indexing/build_retrieval_artifacts.py --preset close_ended_dev_final
```

Attack presets are stricter. They require the attacked corpus to be a clean
prefix plus an appended malicious suffix, and they append only the new chunks
and embeddings to a clean retrieval base.

Typical attack builds:

```powershell
python scripts/indexing/build_retrieval_artifacts.py --preset instruction_following_close_ended_main
python scripts/indexing/build_retrieval_artifacts.py --preset exfiltration_close_ended_main
python scripts/indexing/build_retrieval_artifacts.py --preset exfiltration_close_ended_main_control
python scripts/indexing/build_retrieval_artifacts.py --preset poisoned_rag_close_ended_main
```

For `close_ended_main`, attack presets append to the shared `clean_complete`
retrieval base. For `close_ended_dev`, attack presets append to the matching
`close_ended_dev_final` clean retrieval base.

## Preset Inventory

The repository currently exposes these named presets through
`scripts/indexing/indexing_presets.py`:

- `clean_complete`
- `close_ended_dev_initial`
- `close_ended_dev_final`
- `instruction_following_close_ended_dev`
- `instruction_following_close_ended_main`
- `exfiltration_close_ended_dev`
- `exfiltration_close_ended_dev_control`
- `exfiltration_close_ended_main`
- `exfiltration_close_ended_main_control`
- `poisoned_rag_close_ended_dev`
- `poisoned_rag_close_ended_main`

There are no `close_ended_main_initial` or `close_ended_main_final` indexing
presets because clean `close_ended_main` runs reuse `clean_complete`.

## Compatibility Rules

When you build attack retrieval artifacts, these settings are structural
compatibility inputs and must match the clean retrieval base:

- `--tokenizer-name`
- `--chunk-size`
- `--overlap`
- the effective corpus content and order

The attack builder validates:

- the attacked corpus is a strict clean-prefix extension
- the attack output paths do not overwrite the clean base artifacts
- persisted clean-base build settings match the requested attack-build settings
  when `retrieval_build_info.json` is present

`--device` and `--batch-size` are performance settings rather than structural
compatibility settings. They can change runtime and may affect bit-level
reproducibility, but they should not change corpus structure or chunk identity.

## Common Flags

`build_retrieval_artifacts.py` accepts:

- `--preset`: named preset from `indexing_presets.py`
- `--input-path`: optional corpus override
- `--chunks-path`: optional chunks output override
- `--index-path`: optional FAISS output override
- `--meta-path`: optional metadata output override
- `--tokenizer-name`: tokenizer used for chunk boundaries
- `--chunk-size`
- `--overlap`
- `--device`: `auto`, `cpu`, or `cuda`
- `--batch-size`

Use output-path overrides carefully. For attack builds they must point to
attack-specific artifact paths, not to the clean retrieval base.

## Output Files

Each successful build writes:

- `chunks.jsonl`
- `faiss.index`
- `faiss_meta.json`
- `retrieval_build_info.json`

`retrieval_build_info.json` is the sidecar file that records the preset, input
paths, chunk settings, device, batch size, and, for attack builds, the clean
retrieval base that the attacked artifacts append to.

## How This Fits the Project Workflow

- Clean data preparation is documented in [../data/README.md](../data/README.md).
- Attack-corpus preparation is documented in [../attacks/README.md](../attacks/README.md).
- Experiment execution is documented in [../run/README.md](../run/README.md).
