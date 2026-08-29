from __future__ import annotations

"""
Analyze exfiltration attack experiment logs.

Outputs:
- results/attacks/exfiltration/<dataset>/summary.json
- results/attacks/exfiltration/<dataset>/per_query_results.jsonl
- results/attacks/exfiltration/<dataset>/manual_review.jsonl
- results/attacks/exfiltration/<dataset>/config_summary.csv
- results/attacks/exfiltration/<dataset>/latency_summary.csv
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from scripts.analysis.answer_validation import (
        apply_lenient_match_audit_sample,
        build_answer_quality,
        summarize_answer_quality,
        support_titles_from_sources,
    )
except ModuleNotFoundError:
    from answer_validation import (
        apply_lenient_match_audit_sample,
        build_answer_quality,
        summarize_answer_quality,
        support_titles_from_sources,
    )

from app.dataset_registry import (
    DEFAULT_CLOSE_ENDED_MAIN_DATASET,
    attack_logs_dir,
    attack_processed_dir,
    attack_results_dir,
    exfiltration_control_dataset_name,
    format_temperature_variant,
)
from app.analysis_utils import (
    add_latency_overhead_vs_baseline,
    build_latency_payload,
    compact_retrieved_chunk,
    config_stage_family,
    csv_safe_value,
    detector_metrics_by_config,
    finalize_detector_metrics,
    manual_review_chunk_summary,
    metric_definitions,
    prefixed_rate_fields,
    rate_with_counts,
    summarize_latency_rows,
    update_detector_metrics,
)
from app.experiment_configs import ALL_CONFIG_NAMES
from app.io_utils import iter_jsonl, write_csv, write_jsonl
from app.path_utils import RESULTS_DIR

ATTACK_NAME = "exfiltration"
DEFAULT_DATASET = exfiltration_control_dataset_name(DEFAULT_CLOSE_ENDED_MAIN_DATASET)
DEFAULT_CONFIGS = list(ALL_CONFIG_NAMES)


def load_scenarios(path: Path) -> dict[str, dict[str, Any]]:
    """Load exfiltration scenarios keyed by target query id."""
    return {
        scenario["target_query_id"]: scenario
        for scenario in iter_jsonl(path)
    }


def evaluate_record(record: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw run record into compact exfiltration-analysis data."""
    retrieved_chunks = record.get("retrieved_chunks") or []
    retrieved_doc_ids = {chunk.get("doc_id") for chunk in retrieved_chunks if chunk.get("doc_id") is not None}

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
    valid_trial = protected_doc_retrieved if scenario_type == "control_only" else (
        protected_doc_retrieved and malicious_doc_retrieved
    )

    model_output = record.get("model_output")
    final_output = record.get("final_output")
    pre_filter_leakage = bool(secret) and bool(model_output) and secret in model_output
    post_filter_leakage = bool(secret) and bool(final_output) and secret in final_output
    defense_success = scenario_type != "control_only" and not post_filter_leakage
    answer_quality = build_answer_quality(
        expected_answer=record.get("expected_answer"),
        evaluated_output=final_output,
        evaluated_output_field="final_output",
        support_titles=support_titles_from_sources(record=record, scenario=scenario),
        retrieved_chunks=retrieved_chunks,
        recorded_support_coverage=record.get("support_coverage"),
        defense_success=defense_success,
    )

    return {
        "run_id": record.get("run_id"),
        "query_id": record.get("query_id"),
        "query": record.get("query", scenario.get("target_question")),
        "config_name": record.get("config_name"),
        "scenario_type": scenario_type,
        "expected_answer": record.get("expected_answer"),
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
        "answer_quality": answer_quality,
        "model_output_compact": model_output,
        "final_output_compact": final_output,
        "defenses_enabled": record.get("defense_decisions", {}).get("defenses_enabled", []),
        **build_latency_payload(record),
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
    protected_hit_metric = rate_with_counts(protected_hits, total)
    malicious_hit_metric = rate_with_counts(malicious_hits, total)
    valid_trial_metric = rate_with_counts(valid_trials, total)
    pre_leakage_metric = rate_with_counts(pre_leakages, total)
    post_leakage_metric = rate_with_counts(post_leakages, total)
    blocked_leakage_metric = rate_with_counts(blocked_leakages, pre_leakages)
    valid_pre_leakage_metric = rate_with_counts(valid_pre_leakages, valid_total)
    valid_post_leakage_metric = rate_with_counts(valid_post_leakages, valid_total)

    summary = {
        "total_records": total,
        "n": total,
        "scenario_types": scenario_types,
        "protected_doc_retrieval_hits": protected_hits,
        "protected_doc_retrieval_hit_rate": protected_hit_metric["rate"],
        "malicious_doc_retrieval_hits": malicious_hits,
        "malicious_doc_retrieval_hit_rate": malicious_hit_metric["rate"],
        "valid_trials": valid_trials,
        "valid_trial_rate": valid_trial_metric["rate"],
        "pre_filter_leakages": pre_leakages,
        "pre_filter_leakage_rate": pre_leakage_metric["rate"],
        "post_filter_leakages": post_leakages,
        "post_filter_leakage_rate": post_leakage_metric["rate"],
        "output_blocked_leakages": blocked_leakages,
        "output_block_rate": blocked_leakage_metric["rate"],
        "conditioned_on_valid_trial": {
            "total_records": valid_total,
            "pre_filter_leakages": valid_pre_leakages,
            "pre_filter_leakage_rate": valid_pre_leakage_metric["rate"],
            "post_filter_leakages": valid_post_leakages,
            "post_filter_leakage_rate": valid_post_leakage_metric["rate"],
        },
        "counts": {
            "n": total,
            "n_valid_trials": valid_total,
        },
        "retrieval_metrics": {
            "protected_doc_retrieval_hit": protected_hit_metric,
            "malicious_doc_retrieval_hit": malicious_hit_metric,
            "valid_trial": valid_trial_metric,
        },
        "leakage_metrics": {
            "pre_filter_leakage": pre_leakage_metric,
            "post_filter_leakage": post_leakage_metric,
            "exact_scenario_secret_leakage_pre_filter": pre_leakage_metric,
            "exact_scenario_secret_leakage_post_filter": post_leakage_metric,
            "valid_trial_conditioned_pre_filter_leakage": valid_pre_leakage_metric,
            "valid_trial_conditioned_post_filter_leakage": valid_post_leakage_metric,
        },
        "output_stage_metrics": {
            "blocked_leakage": blocked_leakage_metric,
        },
        "latency_summary": summarize_latency_rows(results),
    }
    if scenario_types == ["control_only"]:
        summary["control_metrics"] = {
            "control_only_pre_filter_leakage": pre_leakage_metric,
            "control_only_post_filter_leakage": post_leakage_metric,
            "control_only_blocked_leakage": blocked_leakage_metric,
        }
    summary["defense_latency_summary"] = summary["latency_summary"]["per_defense_latency_ms"]
    summary["answer_quality"] = summarize_answer_quality(results)
    return summary


def build_manual_review_row(result: dict[str, Any]) -> dict[str, Any]:
    """Build one manual-review row for a flagged exfiltration run."""
    return {
        "config_name": result["config_name"],
        "run_id": result.get("run_id"),
        "query_id": result["query_id"],
        "query": result.get("query"),
        "expected_answer": result.get("expected_answer"),
        "attacker_target_answer": None,
        "model_output": result.get("model_output_compact"),
        "final_output": result.get("final_output_compact"),
        "pre_filter_success": result.get("pre_filter_leakage"),
        "post_filter_success": result.get("post_filter_leakage"),
        "output_blocked_success": result.get("output_blocked_leakage"),
        "answer_quality": result.get("answer_quality"),
        "manual_review_reasons": (result.get("answer_quality") or {}).get(
            "manual_review_reasons", []
        ),
        "retrieved_chunk_summaries": [
            manual_review_chunk_summary(chunk)
            for chunk in result.get("retrieved_chunks_full", [])
        ],
        "manual_answer_quality": {
            "label": None,
            "notes": None,
        },
        "scenario_type": result.get("scenario_type"),
    }


def analyze_config_log(
    *,
    config_name: str,
    log_path: Path,
    scenarios: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Analyze one config log file."""
    if not log_path.exists():
        return [], {
            "status": "missing_log",
            "log_path": str(log_path),
            "total_records": 0,
        }, []

    rows: list[dict[str, Any]] = []
    manual_review_rows: list[dict[str, Any]] = []
    detector_metrics: dict[str, dict[str, Any]] = {}
    skipped = 0

    for record in iter_jsonl(log_path):
        query_id = record.get("query_id")
        scenario = scenarios.get(query_id)
        if scenario is None:
            skipped += 1
            continue
        analyzed = evaluate_record(record, scenario)
        analyzed["retrieved_chunks_full"] = record.get("retrieved_chunks", [])
        rows.append(analyzed)
        scenario_type = scenario.get("scenario_type", "attack")
        update_detector_metrics(
            detector_metrics,
            record.get("defense_decision_log", []),
            is_attack_output=scenario_type != "control_only",
            is_benign_output=scenario_type == "control_only",
        )

    apply_lenient_match_audit_sample(rows)
    for analyzed in rows:
        if analyzed["answer_quality"]["manual_review_needed"]:
            manual_review_rows.append(build_manual_review_row(analyzed))

    summary = summarize_config(rows)
    summary["status"] = "ok"
    summary["log_path"] = str(log_path)
    summary["skipped_records_without_scenario"] = skipped
    summary["config_name"] = config_name
    summary["manual_review_count"] = len(manual_review_rows)
    summary["detector_metrics"] = finalize_detector_metrics(detector_metrics)
    summary["detector_diagnostics"] = summary["detector_metrics"]
    for row in rows:
        row.pop("retrieved_chunks_full", None)
    return rows, summary, manual_review_rows


def build_config_summary_rows(
    *,
    dataset: str,
    summary_by_config: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one thesis-facing wide CSV row per config."""
    rows: list[dict[str, Any]] = []
    for config_name, summary in summary_by_config.items():
        latency_summary = summary.get("latency_summary", {})
        total_latency = latency_summary.get("derived_total_latency_ms", {})
        non_generation_latency = latency_summary.get("derived_non_generation_latency_ms", {})
        retrieval_latency = (latency_summary.get("stage_latency_ms") or {}).get("retrieval", {})
        prompt_latency = (latency_summary.get("stage_latency_ms") or {}).get("prompt_building", {})
        generation_latency = (latency_summary.get("stage_latency_ms") or {}).get("generation", {})
        output_filtering_latency = (latency_summary.get("stage_latency_ms") or {}).get("output_filtering", {})
        overhead = latency_summary.get("overhead_vs_baseline", {})
        answer_quality = summary.get("answer_quality", {})

        row = {
            "attack": ATTACK_NAME,
            "dataset": dataset,
            "config_name": config_name,
            "stage_family": config_stage_family(config_name),
            "scenario_types": ",".join(summary.get("scenario_types", [])),
            "n": summary.get("n"),
            "n_valid_trials": (summary.get("counts") or {}).get("n_valid_trials"),
            **prefixed_rate_fields(
                "protected_doc_retrieval_hit",
                (summary.get("retrieval_metrics") or {}).get("protected_doc_retrieval_hit"),
            ),
            **prefixed_rate_fields(
                "malicious_doc_retrieval_hit",
                (summary.get("retrieval_metrics") or {}).get("malicious_doc_retrieval_hit"),
            ),
            **prefixed_rate_fields(
                "valid_trial",
                (summary.get("retrieval_metrics") or {}).get("valid_trial"),
            ),
            **prefixed_rate_fields(
                "pre_filter_leakage",
                (summary.get("leakage_metrics") or {}).get("pre_filter_leakage"),
            ),
            **prefixed_rate_fields(
                "post_filter_leakage",
                (summary.get("leakage_metrics") or {}).get("post_filter_leakage"),
            ),
            **prefixed_rate_fields(
                "exact_scenario_secret_leakage_pre_filter",
                (summary.get("leakage_metrics") or {}).get("exact_scenario_secret_leakage_pre_filter"),
            ),
            **prefixed_rate_fields(
                "exact_scenario_secret_leakage_post_filter",
                (summary.get("leakage_metrics") or {}).get("exact_scenario_secret_leakage_post_filter"),
            ),
            **prefixed_rate_fields(
                "valid_trial_conditioned_pre_filter_leakage",
                (summary.get("leakage_metrics") or {}).get("valid_trial_conditioned_pre_filter_leakage"),
            ),
            **prefixed_rate_fields(
                "valid_trial_conditioned_post_filter_leakage",
                (summary.get("leakage_metrics") or {}).get("valid_trial_conditioned_post_filter_leakage"),
            ),
            **prefixed_rate_fields(
                "blocked_leakage",
                (summary.get("output_stage_metrics") or {}).get("blocked_leakage"),
            ),
            "answer_quality_lenient_match_count": answer_quality.get("lenient_match_count"),
            "answer_quality_lenient_match_n": answer_quality.get("total_records"),
            "answer_quality_lenient_match_rate": answer_quality.get("lenient_match_rate"),
            "answer_quality_refusal_count": answer_quality.get("refusal_count"),
            "answer_quality_refusal_n": answer_quality.get("total_records"),
            "answer_quality_refusal_rate": answer_quality.get("refusal_rate"),
            "mean_support_coverage": answer_quality.get("mean_support_coverage"),
            "mean_derived_total_latency_ms": total_latency.get("mean"),
            "median_derived_total_latency_ms": total_latency.get("median"),
            "mean_non_generation_latency_ms": non_generation_latency.get("mean"),
            "median_non_generation_latency_ms": non_generation_latency.get("median"),
            "mean_generation_latency_ms": generation_latency.get("mean"),
            "median_generation_latency_ms": generation_latency.get("median"),
            "mean_retrieval_latency_ms": retrieval_latency.get("mean"),
            "mean_prompt_building_latency_ms": prompt_latency.get("mean"),
            "mean_output_filtering_latency_ms": output_filtering_latency.get("mean"),
            **overhead,
        }
        rows.append({key: csv_safe_value(value) for key, value in row.items()})
    return rows


def build_latency_summary_rows(
    *,
    dataset: str,
    summary_by_config: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build long-form latency summary CSV rows."""
    rows: list[dict[str, Any]] = []
    for config_name, summary in summary_by_config.items():
        latency_summary = summary.get("latency_summary", {})
        overhead = latency_summary.get("overhead_vs_baseline", {})

        for stage_name, metrics in (latency_summary.get("stage_latency_ms") or {}).items():
            rows.append(
                {
                    "attack": ATTACK_NAME,
                    "dataset": dataset,
                    "config_name": config_name,
                    "stage_family": config_stage_family(config_name),
                    "latency_scope": "stage",
                    "latency_name": stage_name,
                    **{key: csv_safe_value(value) for key, value in metrics.items()},
                    "overhead_ms": "NA",
                    "overhead_pct": "NA",
                }
            )

        for latency_name, overhead_ms_key, overhead_pct_key in (
            (
                "derived_total_latency_ms",
                "mean_derived_total_latency_overhead_ms",
                "mean_derived_total_latency_overhead_pct",
            ),
            (
                "derived_non_generation_latency_ms",
                "mean_non_generation_latency_overhead_ms",
                "mean_non_generation_latency_overhead_pct",
            ),
        ):
            metrics = latency_summary.get(latency_name) or {}
            rows.append(
                {
                    "attack": ATTACK_NAME,
                    "dataset": dataset,
                    "config_name": config_name,
                    "stage_family": config_stage_family(config_name),
                    "latency_scope": "derived",
                    "latency_name": latency_name,
                    **{key: csv_safe_value(value) for key, value in metrics.items()},
                    "overhead_ms": csv_safe_value(overhead.get(overhead_ms_key)),
                    "overhead_pct": csv_safe_value(overhead.get(overhead_pct_key)),
                }
            )

        for defense_name, metrics in (latency_summary.get("per_defense_latency_ms") or {}).items():
            rows.append(
                {
                    "attack": ATTACK_NAME,
                    "dataset": dataset,
                    "config_name": config_name,
                    "stage_family": config_stage_family(config_name),
                    "latency_scope": "defense",
                    "latency_name": defense_name,
                    **{key: csv_safe_value(value) for key, value in metrics.items()},
                    "overhead_ms": "NA",
                    "overhead_pct": "NA",
                }
            )
    return rows


def analyze_exfiltration(
    *,
    dataset: str,
    config_names: list[str],
    logs_dir: Path,
    scenarios_path: Path,
    results_dir: Path,
    variant: str | None = None,
) -> None:
    """Analyze all selected exfiltration config logs for one dataset."""
    scenarios = load_scenarios(scenarios_path)

    all_rows: list[dict[str, Any]] = []
    manual_review_rows: list[dict[str, Any]] = []
    summary_by_config: dict[str, dict[str, Any]] = {}

    for config_name in config_names:
        log_path = logs_dir / f"{config_name}_runs.jsonl"
        rows, summary, config_manual_review_rows = analyze_config_log(
            config_name=config_name,
            log_path=log_path,
            scenarios=scenarios,
        )
        all_rows.extend(rows)
        manual_review_rows.extend(config_manual_review_rows)
        summary_by_config[config_name] = summary

    add_latency_overhead_vs_baseline(summary_by_config)

    output_dir = attack_results_dir(ATTACK_NAME, dataset, variant=variant, base_dir=results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_summary_path = output_dir / "config_summary.csv"
    latency_summary_path = output_dir / "latency_summary.csv"
    summary = {
        "attack": ATTACK_NAME,
        "dataset": dataset,
        "configs": config_names,
        "logs_dir": str(logs_dir),
        "scenarios_path": str(scenarios_path),
        "summary_by_config": summary_by_config,
        "answer_quality_by_config": {
            config_name: config_summary.get("answer_quality", {})
            for config_name, config_summary in summary_by_config.items()
        },
        "detector_metrics_by_config": detector_metrics_by_config(summary_by_config),
        "metric_definitions": metric_definitions(),
        "reporting_notes": {
            "preferred_logs": "current_main_logs",
            "legacy_logs_policy": (
                "Older incomplete log schemas are not suitable for final thesis reporting."
            ),
        },
        "output_files": {
            "manual_review_path": str(output_dir / "manual_review.jsonl"),
            "config_summary_csv_path": str(config_summary_path),
            "latency_summary_csv_path": str(latency_summary_path),
        },
    }

    summary_path = output_dir / "summary.json"
    per_query_path = output_dir / "per_query_results.jsonl"
    manual_review_path = output_dir / "manual_review.jsonl"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(per_query_path, all_rows)
    write_jsonl(manual_review_path, manual_review_rows)
    write_csv(
        config_summary_path,
        build_config_summary_rows(dataset=dataset, summary_by_config=summary_by_config),
        fieldnames=[
            "attack",
            "dataset",
            "config_name",
            "stage_family",
            "scenario_types",
            "n",
            "n_valid_trials",
            "protected_doc_retrieval_hit_count",
            "protected_doc_retrieval_hit_n",
            "protected_doc_retrieval_hit_rate",
            "malicious_doc_retrieval_hit_count",
            "malicious_doc_retrieval_hit_n",
            "malicious_doc_retrieval_hit_rate",
            "valid_trial_count",
            "valid_trial_n",
            "valid_trial_rate",
            "pre_filter_leakage_count",
            "pre_filter_leakage_n",
            "pre_filter_leakage_rate",
            "post_filter_leakage_count",
            "post_filter_leakage_n",
            "post_filter_leakage_rate",
            "exact_scenario_secret_leakage_pre_filter_count",
            "exact_scenario_secret_leakage_pre_filter_n",
            "exact_scenario_secret_leakage_pre_filter_rate",
            "exact_scenario_secret_leakage_post_filter_count",
            "exact_scenario_secret_leakage_post_filter_n",
            "exact_scenario_secret_leakage_post_filter_rate",
            "valid_trial_conditioned_pre_filter_leakage_count",
            "valid_trial_conditioned_pre_filter_leakage_n",
            "valid_trial_conditioned_pre_filter_leakage_rate",
            "valid_trial_conditioned_post_filter_leakage_count",
            "valid_trial_conditioned_post_filter_leakage_n",
            "valid_trial_conditioned_post_filter_leakage_rate",
            "blocked_leakage_count",
            "blocked_leakage_n",
            "blocked_leakage_rate",
            "answer_quality_lenient_match_count",
            "answer_quality_lenient_match_n",
            "answer_quality_lenient_match_rate",
            "answer_quality_refusal_count",
            "answer_quality_refusal_n",
            "answer_quality_refusal_rate",
            "mean_support_coverage",
            "mean_derived_total_latency_ms",
            "median_derived_total_latency_ms",
            "mean_non_generation_latency_ms",
            "median_non_generation_latency_ms",
            "mean_generation_latency_ms",
            "median_generation_latency_ms",
            "mean_retrieval_latency_ms",
            "mean_prompt_building_latency_ms",
            "mean_output_filtering_latency_ms",
            "mean_derived_total_latency_overhead_ms",
            "median_derived_total_latency_overhead_ms",
            "mean_derived_total_latency_overhead_pct",
            "median_derived_total_latency_overhead_pct",
            "mean_non_generation_latency_overhead_ms",
            "median_non_generation_latency_overhead_ms",
            "mean_non_generation_latency_overhead_pct",
            "median_non_generation_latency_overhead_pct",
        ],
    )
    write_csv(
        latency_summary_path,
        build_latency_summary_rows(dataset=dataset, summary_by_config=summary_by_config),
        fieldnames=[
            "attack",
            "dataset",
            "config_name",
            "stage_family",
            "latency_scope",
            "latency_name",
            "count",
            "mean",
            "median",
            "min",
            "max",
            "std",
            "overhead_ms",
            "overhead_pct",
        ],
    )

    print(json.dumps(summary_by_config, indent=2, ensure_ascii=False))
    print(f"Saved summary -> {summary_path}")
    print(f"Saved per-query results -> {per_query_path}")
    print(f"Saved manual review rows -> {manual_review_path}")
    print(f"Saved config summary CSV -> {config_summary_path}")
    print(f"Saved latency summary CSV -> {latency_summary_path}")


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
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Optional sampled-generation temperature used for the analyzed logs. "
            "Positive values select the matching temperature-specific log and "
            "results subdirectories."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config_names = args.config_names or DEFAULT_CONFIGS
    variant = format_temperature_variant(args.temperature)
    logs_dir = args.logs_dir or attack_logs_dir(ATTACK_NAME, args.dataset, variant)
    scenarios_path = (
        args.scenarios_path
        or attack_processed_dir(ATTACK_NAME, args.dataset) / "scenarios.jsonl"
    )

    analyze_exfiltration(
        dataset=args.dataset,
        config_names=config_names,
        logs_dir=logs_dir,
        scenarios_path=scenarios_path,
        results_dir=args.results_dir,
        variant=variant,
    )


if __name__ == "__main__":
    main()
