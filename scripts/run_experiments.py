from __future__ import annotations

"""
PyCharm-friendly wrapper for `app.experiment_runner`.

Run this file directly when you want to start a batch experiment from the IDE
without typing command-line flags. Adjust the settings in the configuration
section below, then run the script with PyCharm's normal Run action.

Notes:
- `baseline`, `retrieval_only`, `prompt_only`, and `output_only` are executable
"""

from pathlib import Path

from app.experiment_runner import run_experiments


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
LOG_DIR = PROJECT_DIR / "logs"


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
# Recommended current workflow:
# - use "dev" for smoke tests and debugging
# - use "main" for the real baseline runs
# - use "validation" later for a smaller confirmatory subset
PROFILE_NAME = "dev"

# Currently safe choices:
# - ["baseline"]
# - ["retrieval_only"]
# - ["prompt_only"]
# - ["output_only"]
CONFIG_NAMES = ["output_only"]

# Set to an integer for a small debug run, or `None` for the full query file.
LIMIT = 10

QUERIES_PATH = DATA_DIR / "processed" / "hotpot_queries_dev.jsonl"
INDEX_PATH = DATA_DIR / "built" / "faiss_dev.index"
META_PATH = DATA_DIR / "built" / "faiss_dev_meta.json"

# Change the filename when you want to keep multiple experiment batches.
BASELINE_DEV_OUTPUT_PATH = LOG_DIR / "baseline_dev_runs.jsonl"
RETRIEVAL_ONLY_DEV_OUTPUT_PATH = LOG_DIR / "retrieval_only_dev_runs.jsonl"
PROMPT_ONLY_DEV_OUTPUT_PATH = LOG_DIR / "prompt_only_dev_runs.jsonl"
OUTPUT_ONLY_DEV_OUTPUT_PATH = LOG_DIR / "output_only_dev_runs.jsonl"



OUTPUT_PATH = OUTPUT_ONLY_DEV_OUTPUT_PATH

def main() -> None:
    """Run the configured experiment batch."""
    run_experiments(
        queries_path=QUERIES_PATH,
        index_path=INDEX_PATH,
        meta_path=META_PATH,
        output_path=OUTPUT_PATH,
        config_names=CONFIG_NAMES,
        profile_name=PROFILE_NAME,
        limit=LIMIT,
    )


if __name__ == "__main__":
    main()
