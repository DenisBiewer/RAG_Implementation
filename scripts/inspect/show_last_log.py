import json

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = next(
    parent
    for parent in (BASE_DIR, *BASE_DIR.parents)
    if (parent / "app").exists() and (parent / "data").exists()
)

LOG_PATH = PROJECT_DIR / "logs" / "clean" / "dev" / "baseline_runs.jsonl"

def main():
    with open(str(LOG_PATH), "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("No log records found.")
        return

    last = json.loads(lines[-1])

    print("RUN ID:", last["run_id"])
    print("QUERY:", last["query"])
    print()
    print("OUTPUT:", last["model_output"])
    print("LATENCIES:", last["latencies_ms"])


if __name__ == "__main__":
    main()
