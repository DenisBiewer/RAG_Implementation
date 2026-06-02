from __future__ import annotations

from typing import Any


def unique_support_titles(query_record: dict[str, Any]) -> list[str]:
    """Return unique supporting titles while preserving their original order."""
    supporting_facts = query_record.get("supporting_facts") or {}
    titles = supporting_facts.get("title") or []
    return list(dict.fromkeys(title for title in titles if title))
