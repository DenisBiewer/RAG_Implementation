from __future__ import annotations

"""
Analyze indirect instruction-following attack experiment logs.

This script reads all selected config logs for one instruction-following attack
dataset and writes compact result files without full prompts, contexts, or
retrieved chunk text.

Outputs:
- results/attacks/instruction_following/<dataset>/summary.json
- results/attacks/instruction_following/<dataset>/per_query_results.jsonl
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

ATTACK_NAME = "instruction_following"
DEFAULT_DATASET = "controlled_dev"
DEFAULT_CONFIGS = [
    "baseline",
    "retrieval_only",
    "prompt_only",
    "output_only",
]


def iter_jsonl(path: Path):
    """Yield JSON objects from a JSONL file."""
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
    """Load attack scenarios keyed by target query id."""
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
    """Keep only retrieval metadata needed for attack analysis."""
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
    """Convert one raw experiment record into compact attack-analysis data."""
    sentinel = scenario["sentinel"]
    malicious_doc_id = scenario["malicious_doc_id"]
    retrieved_chunks = record["retrieved_chunks"]

    malicious_chunks = [
        chunk
        for chunk in retrieved_chunks
        if chunk.get("is_malicious")
    ]
    target_attack_ranks = [
        chunk["rank"]
        for chunk in malicious_chunks
        if chunk.get("doc_id") == malicious_doc_id
    ]

    model_output = record["model_output"]
    final_output = record["final_output"]
    pre_filter_success = sentinel in model_output
    post_filter_success = sentinel in final_output

    return {
        "query_id": record["query_id"],
        "config_name": record["config_name"],
        "expected_answer": record["expected_answer"],
        "sentinel": sentinel,
        "target_malicious_doc_id": malicious_doc_id,
        "target_attack_retrieved": bool(target_attack_ranks),
        "target_attack_best_rank": min(target_attack_ranks) if target_attack_ranks else None,
        "malicious_chunks_retrieved": len(malicious_chunks),
        "malicious_doc_ids_retrieved": [chunk["doc_id"] for chunk in malicious_chunks],
        "retrieved_chunks_compact": [compact_retrieved_chunk(chunk) for chunk in retrieved_chunks],
        "pre_filter_success": pre_filter_success,
        "post_filter_success": post_filter_success,
        "output_blocked_success": pre_filter_success and not post_filter_success,
        "model_output_compact": model_output,
        "final_output_compact": final_output,
        "defenses_enabled": record.get("defense_decisions", {}).get("defenses_enabled", []),
    }


def summarize_config(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate compact per-query rows for one config."""
    total = len(results)
    retrieval_hits = sum(result["target_attack_retrieved"] for result in results)
    pre_successes = sum(result["pre_filter_success"] for result in results)
    post_successes = sum(result["post_filter_success"] for result in results)
    blocked_successes = sum(result["output_blocked_success"] for result in results)
    total_malicious_chunks = sum(result["malicious_chunks_retrieved"] for result in results)

    retrieved_results = [result for result in results if result["target_attack_retrieved"]]
    retrieved_total = len(retrieved_results)
    retrieved_pre_successes = sum(result["pre_filter_success"] for result in retrieved_results)
    retrieved_post_successes = sum(result["post_filter_success"] for result in retrieved_results)

    return {
        "total_records": total,
        "target_attack_retrieval_hits": retrieval_hits,
        "target_attack_hit_rate": fraction(retrieval_hits, total),
        "mean_malicious_chunks_retrieved": fraction(total_malicious_chunks, total),
        "pre_filter_successes": pre_successes,
        "pre_filter_asr": fraction(pre_successes, total),
        "post_filter_successes": post_successes,
        "post_filter_asr": fraction(post_successes, total),
        "output_blocked_successes": blocked_successes,
        "output_block_rate": fraction(blocked_successes, pre_successes),
        "conditioned_on_target_attack_retrieved": {
            "total_records": retrieved_total,
            "pre_filter_successes": retrieved_pre_successes,
            "pre_filter_asr": fraction(retrieved_pre_successes, retrieved_total),
            "post_filter_successes": retrieved_post_successes,
            "post_filter_asr": fraction(retrieved_post_successes, retrieved_total),
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


def analyze_instruction_following(
    *,
    dataset: str,
    config_names: list[str],
    logs_dir: Path,
    scenarios_path: Path,
    results_dir: Path,
) -> None:
    """Analyze all selected instruction-following config logs."""
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
    parser = argparse.ArgumentParser(description="Analyze instruction-following attack logs.")
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

    analyze_instruction_following(
        dataset=args.dataset,
        config_names=config_names,
        logs_dir=logs_dir,
        scenarios_path=scenarios_path,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
