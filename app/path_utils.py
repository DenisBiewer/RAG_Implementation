from __future__ import annotations

"""Shared project path constants."""

from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
BUILT_DIR = DATA_DIR / "built"
LOG_DIR = PROJECT_DIR / "logs"
RESULTS_DIR = PROJECT_DIR / "results"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
