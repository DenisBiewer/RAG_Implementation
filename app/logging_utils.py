from __future__ import annotations

import time
import uuid

from app.io_utils import append_jsonl


def new_run_id() -> str:
    return str(uuid.uuid4())


class StageTimer:
    """
    Utility for simple per-stage latency recording in milliseconds.
    """

    def __init__(self):
        self.timings: dict[str, float] = {}

    def start(self, stage_name: str) -> float:
        return time.perf_counter()

    def stop(self, stage_name: str, start_time: float) -> None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.timings[stage_name] = round(elapsed_ms, 3)
