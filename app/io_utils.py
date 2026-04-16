from __future__ import annotations

"""Helpers for simple filesystem operations used across the application.

The module currently focuses on JSON Lines (JSONL) input/output and on making
sure destination directories exist before files are written.
"""

import json
from pathlib import Path
from typing import Iterable


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory for ``path`` if it does not already exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    """Write records to a UTF-8 encoded JSONL file.

    Each item in ``records`` is serialized as one JSON object per line. Missing
    parent directories are created automatically.
    """
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file and return its parsed records.

    Blank lines are ignored. Each non-empty line is expected to contain a valid
    JSON object.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
