from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Callable, Iterable, TypeVar

from app.seed_config import derive_seed


RecordT = TypeVar("RecordT")


def stable_hash_rank(
    seed: int,
    record_id: object,
    namespace: str,
    *parts: object,
) -> int:
    """Return a stable seeded integer rank for one record."""
    return derive_seed(seed, namespace, record_id, *parts)


def stable_rank_key(
    seed: int,
    record_id: object,
    namespace: str,
    *parts: object,
) -> tuple[int, str]:
    """Return a comparable seeded ranking key with deterministic tie-breaking."""
    normalized_record_id = str(record_id)
    return (
        stable_hash_rank(seed, normalized_record_id, namespace, *parts),
        normalized_record_id,
    )


def select_seeded_subset(
    records: list[RecordT],
    *,
    count: int,
    seed: int,
    id_fn: Callable[[RecordT], object],
    namespace: str,
) -> tuple[list[RecordT], list[RecordT]]:
    """Select a stable seeded subset from an in-memory record list."""
    ranked_records = sorted(
        records,
        key=lambda record: stable_rank_key(seed, id_fn(record), namespace),
    )
    return ranked_records[:count], ranked_records[count:]


@dataclass(frozen=True)
class _MaxHeapKey:
    """Reverse-ordered key wrapper so heap[0] is the current worst selected row."""

    rank: int
    record_id: str

    def __lt__(self, other: _MaxHeapKey) -> bool:
        return (self.rank, self.record_id) > (other.rank, other.record_id)


def select_seeded_top_k(
    records: Iterable[RecordT],
    *,
    count: int,
    seed: int,
    id_fn: Callable[[RecordT], object],
    namespace: str,
) -> tuple[list[RecordT], int]:
    """
    Select the best seeded top-k records from a stream without storing all rows.

    Returns the selected rows in ascending seeded-rank order plus the total
    candidate count seen in the stream.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}.")

    if count == 0:
        return [], sum(1 for _ in records)

    heap: list[tuple[_MaxHeapKey, int, RecordT]] = []
    candidate_count = 0

    for candidate_count, record in enumerate(records, start=1):
        record_id = str(id_fn(record))
        rank, normalized_record_id = stable_rank_key(seed, record_id, namespace)
        heap_item = (_MaxHeapKey(rank=rank, record_id=normalized_record_id), candidate_count, record)

        if len(heap) < count:
            heapq.heappush(heap, heap_item)
            continue

        worst_key = heap[0][0]
        if (rank, normalized_record_id) < (worst_key.rank, worst_key.record_id):
            heapq.heapreplace(heap, heap_item)

    selected_records = [
        item[2]
        for item in sorted(heap, key=lambda item: (item[0].rank, item[0].record_id))
    ]
    return selected_records, candidate_count
