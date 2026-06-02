from __future__ import annotations

"""Helpers for simple filesystem operations used across the application.

The module currently focuses on JSON and JSON Lines (JSONL) input/output and
on making sure destination directories exist before files are written.
"""

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory for ``path`` if it does not already exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one record to a UTF-8 encoded JSONL file."""
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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
    return list(iter_jsonl(path))


def write_json(path: str | Path, data: Any, *, indent: int | None = 2) -> None:
    """Write one JSON document to disk."""
    ensure_parent_dir(path)
    Path(path).write_text(
        json.dumps(data, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> None:
    """Write a UTF-8 CSV file with a fixed column order."""
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: str | Path) -> Any:
    """Read one JSON document from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
