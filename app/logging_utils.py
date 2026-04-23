from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


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


def append_jsonl(path: str, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")