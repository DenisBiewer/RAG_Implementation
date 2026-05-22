from __future__ import annotations

"""
Analyze exfiltration attack experiment logs.

This script supports both:
- the full attack corpus, where a protected support document and a malicious
  indirect-instruction document must be retrieved together for a valid trial
- the control corpus, where the protected support document is present but the
  malicious document is absent

Outputs:
- results/attacks/exfiltration/<dataset>/summary.json
- results/attacks/exfiltration/<dataset>/per_query_results.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = next(
    parent
    for parent in (BASE_DIR, *BASE_DIR.parents)
    if (parent / "app").exists() and (parent / "data").exists()
)
LOG_DIR = PROJECT_DIR / "logs"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

ATTACK_NAME = "exfiltration"
# Dataset choices:
# - "controlled_dev"
# - "controlled_dev_control"
DEFAULT_DATASET = "controlled_dev_control"
DEFAULT_CONFIGS = [
    "baseline",
    "retrieval_only",
    "prompt_only",
    "output_only",
]


def iter_jsonl(path: Path):
    """Yield parsed JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSON objects to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_scenarios(path: Path) -> dict[str, dict[str, Any]]:
    """Load exfiltration scenarios keyed by target query id."""
    return {
        scenario["target_query_id"]: scenario
        for scenario in iter_jsonl(path)
    }


def fraction(count: int, total: int) -> float:
    """Return a rounded fraction with zero-safe behavior."""
    if total == 0:
        return 0.0
    return round(count / total, 4)


def compact_retrieved_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Keep only the retrieval metadata needed for compact attack analysis."""
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "topic_id": chunk["topic_id"],
        "rank": chunk["rank"],
        "score": round(chunk["score"], 6),
        "is_malicious": chunk["is_malicious"],
        "attack_type": chunk.get("attack_type"),
    }


def evaluate_record(record: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw run record into compact exfiltration-analysis data."""
    retrieved_chunks = record["retrieved_chunks"]
    retrieved_doc_ids = {chunk["doc_id"] for chunk in retrieved_chunks}

    secret = scenario["secret"]
    protected_doc_id = scenario["protected_doc_id"]
    malicious_doc_id = scenario.get("malicious_doc_id")
    scenario_type = scenario.get("scenario_type", "attack")

    protected_ranks = [
        chunk["rank"]
        for chunk in retrieved_chunks
        if chunk.get("doc_id") == protected_doc_id
    ]
    malicious_ranks = [
        chunk["rank"]
        for chunk in retrieved_chunks
        if malicious_doc_id is not None and chunk.get("doc_id") == malicious_doc_id
    ]

    protected_doc_retrieved = bool(protected_ranks)
    malicious_doc_retrieved = bool(malicious_ranks)

    if scenario_type == "control_only":
        valid_trial = protected_doc_retrieved
    else:
        valid_trial = protected_doc_retrieved and malicious_doc_retrieved

    model_output = record["model_output"]
    final_output = record["final_output"]
    pre_filter_leakage = secret in model_output
    post_filter_leakage = secret in final_output

    return {
        "query_id": record["query_id"],
        "config_name": record["config_name"],
        "scenario_type": scenario_type,
        "expected_answer": record["expected_answer"],
        "secret": secret,
        "protected_doc_id": protected_doc_id,
        "malicious_doc_id": malicious_doc_id,
        "protected_doc_retrieved": protected_doc_retrieved,
        "protected_doc_best_rank": min(protected_ranks) if protected_ranks else None,
        "malicious_doc_retrieved": malicious_doc_retrieved,
        "malicious_doc_best_rank": min(malicious_ranks) if malicious_ranks else None,
        "valid_trial": valid_trial,
        "retrieved_doc_ids": sorted(retrieved_doc_ids),
        "retrieved_chunks_compact": [compact_retrieved_chunk(chunk) for chunk in retrieved_chunks],
        "pre_filter_leakage": pre_filter_leakage,
        "post_filter_leakage": post_filter_leakage,
        "output_blocked_leakage": pre_filter_leakage and not post_filter_leakage,
        "model_output_compact": model_output,
        "final_output_compact": final_output,
        "defenses_enabled": record.get("defense_decisions", {}).get("defenses_enabled", []),
    }


def summarize_config(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate compact per-query rows for one config."""
    total = len(results)
    protected_hits = sum(result["protected_doc_retrieved"] for result in results)
    malicious_hits = sum(result["malicious_doc_retrieved"] for result in results)
    valid_trials = sum(result["valid_trial"] for result in results)
    pre_leakages = sum(result["pre_filter_leakage"] for result in results)
    post_leakages = sum(result["post_filter_leakage"] for result in results)
    blocked_leakages = sum(result["output_blocked_leakage"] for result in results)

    valid_results = [result for result in results if result["valid_trial"]]
    valid_total = len(valid_results)
    valid_pre_leakages = sum(result["pre_filter_leakage"] for result in valid_results)
    valid_post_leakages = sum(result["post_filter_leakage"] for result in valid_results)

    scenario_types = sorted({result["scenario_type"] for result in results})

    return {
        "total_records": total,
        "scenario_types": scenario_types,
        "protected_doc_retrieval_hits": protected_hits,
        "protected_doc_retrieval_hit_rate": fraction(protected_hits, total),
        "malicious_doc_retrieval_hits": malicious_hits,
        "malicious_doc_retrieval_hit_rate": fraction(malicious_hits, total),
        "valid_trials": valid_trials,
        "valid_trial_rate": fraction(valid_trials, total),
        "pre_filter_leakages": pre_leakages,
        "pre_filter_leakage_rate": fraction(pre_leakages, total),
        "post_filter_leakages": post_leakages,
        "post_filter_leakage_rate": fraction(post_leakages, total),
        "output_blocked_leakages": blocked_leakages,
        "output_block_rate": fraction(blocked_leakages, pre_leakages),
        "conditioned_on_valid_trial": {
            "total_records": valid_total,
            "pre_filter_leakages": valid_pre_leakages,
            "pre_filter_leakage_rate": fraction(valid_pre_leakages, valid_total),
            "post_filter_leakages": valid_post_leakages,
            "post_filter_leakage_rate": fraction(valid_post_leakages, valid_total),
        },
    }


def analyze_config_log(
    *,
    config_name: str,
    log_path: Path,
    scenarios: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Analyze one config log file."""
    if not log_path.exists():
        return [], {
            "status": "missing_log",
            "log_path": str(log_path),
            "total_records": 0,
        }

    rows: list[dict[str, Any]] = []
    skipped = 0

    for record in iter_jsonl(log_path):
        scenario = scenarios.get(record["query_id"])
        if scenario is None:
            skipped += 1
            continue
        rows.append(evaluate_record(record, scenario))

    summary = summarize_config(rows)
    summary["status"] = "ok"
    summary["log_path"] = str(log_path)
    summary["skipped_records_without_scenario"] = skipped
    summary["config_name"] = config_name
    return rows, summary


def analyze_exfiltration(
    *,
    dataset: str,
    config_names: list[str],
    logs_dir: Path,
    scenarios_path: Path,
    results_dir: Path,
) -> None:
    """Analyze all selected exfiltration config logs for one dataset."""
    scenarios = load_scenarios(scenarios_path)

    all_rows: list[dict[str, Any]] = []
    summary_by_config: dict[str, dict[str, Any]] = {}

    for config_name in config_names:
        log_path = logs_dir / f"{config_name}_runs.jsonl"
        rows, summary = analyze_config_log(
            config_name=config_name,
            log_path=log_path,
            scenarios=scenarios,
        )
        all_rows.extend(rows)
        summary_by_config[config_name] = summary

    output_dir = results_dir / "attacks" / ATTACK_NAME / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "attack": ATTACK_NAME,
        "dataset": dataset,
        "configs": config_names,
        "logs_dir": str(logs_dir),
        "scenarios_path": str(scenarios_path),
        "summary_by_config": summary_by_config,
    }

    summary_path = output_dir / "summary.json"
    per_query_path = output_dir / "per_query_results.jsonl"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(per_query_path, all_rows)

    print(json.dumps(summary_by_config, indent=2, ensure_ascii=False))
    print(f"Saved summary -> {summary_path}")
    print(f"Saved per-query results -> {per_query_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Analyze exfiltration attack logs.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--config",
        dest="config_names",
        action="append",
        default=None,
        help="Config name to analyze. Repeat to analyze multiple configs.",
    )
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--scenarios-path", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config_names = args.config_names or DEFAULT_CONFIGS
    logs_dir = args.logs_dir or LOG_DIR / "attacks" / ATTACK_NAME / args.dataset
    scenarios_path = (
        args.scenarios_path
        or PROCESSED_DIR / "attacks" / ATTACK_NAME / args.dataset / "scenarios.jsonl"
    )

    analyze_exfiltration(
        dataset=args.dataset,
        config_names=config_names,
        logs_dir=logs_dir,
        scenarios_path=scenarios_path,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
