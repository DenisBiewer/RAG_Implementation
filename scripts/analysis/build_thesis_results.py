from __future__ import annotations

"""Build cross-attack thesis-facing result tables from analyzer outputs."""

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.analysis_utils import config_stage_family, csv_safe_value, metric_definitions
from app.dataset_registry import DEFAULT_CLOSE_ENDED_MAIN_DATASET
from app.experiment_configs import ALL_CONFIG_NAMES
from app.io_utils import read_json, write_csv, write_json
from app.path_utils import RESULTS_DIR


def load_summary(path: Path, *, required: bool) -> dict[str, Any] | None:
    """Load one analyzer summary file when present."""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required summary is missing: {path}")
        return None
    return read_json(path)


def prefixed_metric(prefix: str, metric: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten one structured rate metric into prefixed fields."""
    metric = metric or {}
    return {
        f"{prefix}_count": metric.get("count"),
        f"{prefix}_n": metric.get("n"),
        f"{prefix}_rate": metric.get("rate"),
    }


def get_metric(summary_by_config: dict[str, Any], config_name: str, *path: str) -> dict[str, Any] | None:
    """Safely resolve one nested metric object from a per-config summary."""
    current = (summary_by_config.get(config_name) or {})
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, dict):
        return current
    return None


def get_value(summary_by_config: dict[str, Any], config_name: str, *path: str) -> Any:
    """Safely resolve one nested scalar from a per-config summary."""
    current: Any = summary_by_config.get(config_name) or {}
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def build_attack_effectiveness_rows(
    dataset: str,
    *,
    instruction_summary: dict[str, Any],
    exfiltration_summary: dict[str, Any],
    poisoned_summary: dict[str, Any],
    exfiltration_control_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build a wide per-config attack effectiveness table."""
    rows: list[dict[str, Any]] = []
    instruction_by_config = instruction_summary.get("summary_by_config", {})
    exfiltration_by_config = exfiltration_summary.get("summary_by_config", {})
    poisoned_by_config = poisoned_summary.get("summary_by_config", {})
    control_by_config = (
        exfiltration_control_summary.get("summary_by_config", {})
        if exfiltration_control_summary is not None
        else {}
    )

    for config_name in ALL_CONFIG_NAMES:
        row = {
            "dataset": dataset,
            "config_name": config_name,
            "stage_family": config_stage_family(config_name),
            **prefixed_metric(
                "instruction_final_asr",
                get_metric(instruction_by_config, config_name, "security_metrics", "final_asr"),
            ),
            **prefixed_metric(
                "instruction_raw_asr",
                get_metric(instruction_by_config, config_name, "security_metrics", "raw_asr"),
            ),
            **prefixed_metric(
                "instruction_retrieval_hit",
                get_metric(instruction_by_config, config_name, "retrieval_metrics", "target_malicious_retrieval_hit"),
            ),
            **prefixed_metric(
                "instruction_blocked_attack_success",
                get_metric(instruction_by_config, config_name, "output_stage_metrics", "blocked_attack_success"),
            ),
            **prefixed_metric(
                "exfiltration_post_filter_leakage",
                get_metric(exfiltration_by_config, config_name, "leakage_metrics", "post_filter_leakage"),
            ),
            **prefixed_metric(
                "exfiltration_pre_filter_leakage",
                get_metric(exfiltration_by_config, config_name, "leakage_metrics", "pre_filter_leakage"),
            ),
            **prefixed_metric(
                "exfiltration_valid_trial",
                get_metric(exfiltration_by_config, config_name, "retrieval_metrics", "valid_trial"),
            ),
            **prefixed_metric(
                "exfiltration_blocked_leakage",
                get_metric(exfiltration_by_config, config_name, "output_stage_metrics", "blocked_leakage"),
            ),
            **prefixed_metric(
                "poisoned_final_rule_asr",
                get_metric(poisoned_by_config, config_name, "attack_metrics", "final_rule_asr"),
            ),
            **prefixed_metric(
                "poisoned_final_substring_asr",
                get_metric(poisoned_by_config, config_name, "attack_metrics", "final_substring_asr"),
            ),
            **prefixed_metric(
                "poisoned_retrieval_hit",
                get_metric(poisoned_by_config, config_name, "retrieval_metrics", "target_poison_retrieval_hit"),
            ),
            **prefixed_metric(
                "poisoned_blocked_rule_success",
                get_metric(poisoned_by_config, config_name, "output_stage_metrics", "blocked_rule_success"),
            ),
            "poisoned_mean_dominance_rate": get_value(
                poisoned_by_config, config_name, "dominance_metrics", "mean_dominance_rate"
            ),
            **prefixed_metric(
                "exfiltration_control_post_filter_leakage",
                get_metric(control_by_config, config_name, "leakage_metrics", "post_filter_leakage"),
            ),
            "notes": (
                "main_exfiltration_control_summary_missing"
                if exfiltration_control_summary is None
                else ""
            ),
        }
        rows.append({key: csv_safe_value(value) for key, value in row.items()})
    return rows


def build_clean_utility_rows(dataset: str, *, clean_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a wide per-config clean utility / false-positive table."""
    rows: list[dict[str, Any]] = []
    summary_by_config = clean_summary.get("summary_by_config", {})
    for config_name in ALL_CONFIG_NAMES:
        row = {
            "dataset": dataset,
            "config_name": config_name,
            "stage_family": config_stage_family(config_name),
            **prefixed_metric(
                "validation_pass",
                get_metric(summary_by_config, config_name, "utility_metrics", "validation_pass"),
            ),
            **prefixed_metric(
                "strict_rule_answer_match",
                get_metric(summary_by_config, config_name, "utility_metrics", "strict_rule_answer_match"),
            ),
            **prefixed_metric(
                "lenient_match",
                get_metric(summary_by_config, config_name, "utility_metrics", "lenient_match"),
            ),
            **prefixed_metric(
                "benign_refusal",
                get_metric(summary_by_config, config_name, "refusal_metrics", "benign_refusal"),
            ),
            **prefixed_metric(
                "cannot_determine",
                get_metric(summary_by_config, config_name, "refusal_metrics", "cannot_determine"),
            ),
            **prefixed_metric(
                "benign_output_false_positive",
                get_metric(summary_by_config, config_name, "false_positive_metrics", "output_false_positive_runs"),
            ),
            **prefixed_metric(
                "retrieval_false_positive_runs",
                get_metric(summary_by_config, config_name, "false_positive_metrics", "retrieval_false_positive_runs"),
            ),
            **prefixed_metric(
                "retrieval_flagged_benign_chunks",
                get_metric(summary_by_config, config_name, "false_positive_metrics", "retrieval_flagged_benign_chunks"),
            ),
            **prefixed_metric(
                "retrieval_downranked_benign_chunks",
                get_metric(summary_by_config, config_name, "false_positive_metrics", "retrieval_downranked_benign_chunks"),
            ),
            **prefixed_metric(
                "prompt_false_positive_runs",
                get_metric(summary_by_config, config_name, "false_positive_metrics", "prompt_false_positive_runs"),
            ),
            **prefixed_metric(
                "benign_docs_labeled_low_trust",
                get_metric(summary_by_config, config_name, "false_positive_metrics", "benign_docs_labeled_low_trust"),
            ),
            **prefixed_metric(
                "benign_docs_labeled_no_trust",
                get_metric(summary_by_config, config_name, "false_positive_metrics", "benign_docs_labeled_no_trust"),
            ),
            "mean_support_coverage": get_value(summary_by_config, config_name, "answer_quality", "mean_support_coverage"),
        }
        rows.append({key: csv_safe_value(value) for key, value in row.items()})
    return rows


def build_latency_overhead_rows(
    dataset: str,
    *,
    clean_summary: dict[str, Any],
    instruction_summary: dict[str, Any],
    exfiltration_summary: dict[str, Any],
    poisoned_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a wide per-config latency and overhead table across analyses."""
    rows: list[dict[str, Any]] = []
    sources = {
        "clean": clean_summary.get("summary_by_config", {}),
        "instruction_following": instruction_summary.get("summary_by_config", {}),
        "exfiltration": exfiltration_summary.get("summary_by_config", {}),
        "poisoned_rag": poisoned_summary.get("summary_by_config", {}),
    }

    for config_name in ALL_CONFIG_NAMES:
        row: dict[str, Any] = {
            "dataset": dataset,
            "config_name": config_name,
            "stage_family": config_stage_family(config_name),
        }
        for source_name, summary_by_config in sources.items():
            latency_summary = get_value(summary_by_config, config_name, "latency_summary") or {}
            total_latency = latency_summary.get("derived_total_latency_ms") or {}
            non_generation_latency = latency_summary.get("derived_non_generation_latency_ms") or {}
            overhead = latency_summary.get("overhead_vs_baseline") or {}
            row.update(
                {
                    f"{source_name}_mean_derived_total_latency_ms": total_latency.get("mean"),
                    f"{source_name}_median_derived_total_latency_ms": total_latency.get("median"),
                    f"{source_name}_mean_non_generation_latency_ms": non_generation_latency.get("mean"),
                    f"{source_name}_median_non_generation_latency_ms": non_generation_latency.get("median"),
                    f"{source_name}_mean_derived_total_latency_overhead_ms": overhead.get(
                        "mean_derived_total_latency_overhead_ms"
                    ),
                    f"{source_name}_median_derived_total_latency_overhead_ms": overhead.get(
                        "median_derived_total_latency_overhead_ms"
                    ),
                    f"{source_name}_mean_derived_total_latency_overhead_pct": overhead.get(
                        "mean_derived_total_latency_overhead_pct"
                    ),
                    f"{source_name}_median_derived_total_latency_overhead_pct": overhead.get(
                        "median_derived_total_latency_overhead_pct"
                    ),
                    f"{source_name}_mean_non_generation_latency_overhead_ms": overhead.get(
                        "mean_non_generation_latency_overhead_ms"
                    ),
                    f"{source_name}_median_non_generation_latency_overhead_ms": overhead.get(
                        "median_non_generation_latency_overhead_ms"
                    ),
                    f"{source_name}_mean_non_generation_latency_overhead_pct": overhead.get(
                        "mean_non_generation_latency_overhead_pct"
                    ),
                    f"{source_name}_median_non_generation_latency_overhead_pct": overhead.get(
                        "median_non_generation_latency_overhead_pct"
                    ),
                }
            )
        rows.append({key: csv_safe_value(value) for key, value in row.items()})
    return rows


def build_compact_overview_rows(
    dataset: str,
    *,
    clean_summary: dict[str, Any],
    instruction_summary: dict[str, Any],
    exfiltration_summary: dict[str, Any],
    poisoned_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build one compact per-config thesis overview row."""
    rows: list[dict[str, Any]] = []
    clean_by_config = clean_summary.get("summary_by_config", {})
    instruction_by_config = instruction_summary.get("summary_by_config", {})
    exfiltration_by_config = exfiltration_summary.get("summary_by_config", {})
    poisoned_by_config = poisoned_summary.get("summary_by_config", {})

    for config_name in ALL_CONFIG_NAMES:
        row = {
            "dataset": dataset,
            "config_name": config_name,
            "stage_family": config_stage_family(config_name),
            **prefixed_metric(
                "instruction_final_asr",
                get_metric(instruction_by_config, config_name, "security_metrics", "final_asr"),
            ),
            **prefixed_metric(
                "exfiltration_post_filter_leakage",
                get_metric(exfiltration_by_config, config_name, "leakage_metrics", "post_filter_leakage"),
            ),
            **prefixed_metric(
                "poisoned_final_rule_asr",
                get_metric(poisoned_by_config, config_name, "attack_metrics", "final_rule_asr"),
            ),
            **prefixed_metric(
                "clean_validation_pass",
                get_metric(clean_by_config, config_name, "utility_metrics", "validation_pass"),
            ),
            **prefixed_metric(
                "clean_benign_refusal",
                get_metric(clean_by_config, config_name, "refusal_metrics", "benign_refusal"),
            ),
            **prefixed_metric(
                "clean_benign_output_false_positive",
                get_metric(clean_by_config, config_name, "false_positive_metrics", "output_false_positive_runs"),
            ),
            "clean_mean_derived_total_latency_overhead_ms": get_value(
                clean_by_config,
                config_name,
                "latency_summary",
                "overhead_vs_baseline",
                "mean_derived_total_latency_overhead_ms",
            ),
            "clean_mean_derived_total_latency_overhead_pct": get_value(
                clean_by_config,
                config_name,
                "latency_summary",
                "overhead_vs_baseline",
                "mean_derived_total_latency_overhead_pct",
            ),
        }
        rows.append({key: csv_safe_value(value) for key, value in row.items()})
    return rows


def build_readme_text(dataset: str, manifest: dict[str, Any]) -> str:
    """Build a short generated README for the thesis output directory."""
    definitions = metric_definitions()
    lines = [
        f"# Thesis Results ({dataset})",
        "",
        "Generated from current analyzer outputs only. Legacy or incomplete development logs are excluded from thesis-facing reporting.",
        "",
        "## Files",
        "- `attack_effectiveness_by_config.csv`: primary attack metrics by configuration.",
        "- `clean_utility_false_positives_by_config.csv`: clean utility, benign refusal, and false-positive metrics by configuration.",
        "- `latency_overhead_by_config.csv`: absolute and baseline-relative latency summaries by configuration.",
        "- `cross_attack_compact_overview.csv`: compact cross-attack overview table.",
        "- `summary.json`: machine-readable combined payload.",
        "- `manifest.json`: source inputs used to build these tables.",
        "",
        "## Metric Definitions",
    ]
    for name, description in definitions.items():
        lines.append(f"- `{name}`: {description}")

    lines.extend(
        [
            "",
            "## Source Inputs",
            f"- Clean summary: `{manifest['inputs']['clean_summary']}`",
            f"- Instruction-following summary: `{manifest['inputs']['instruction_summary']}`",
            f"- Exfiltration summary: `{manifest['inputs']['exfiltration_summary']}`",
            f"- PoisonedRAG summary: `{manifest['inputs']['poisoned_summary']}`",
            f"- Optional exfiltration control summary: `{manifest['inputs']['exfiltration_control_summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_thesis_results(*, dataset: str, results_dir: Path) -> None:
    """Build the final thesis-facing result tables for one dataset."""
    validation_summary_path = results_dir / "validation" / dataset / "summary.json"
    instruction_summary_path = results_dir / "attacks" / "instruction_following" / dataset / "summary.json"
    exfiltration_summary_path = results_dir / "attacks" / "exfiltration" / dataset / "summary.json"
    poisoned_summary_path = results_dir / "attacks" / "poisoned_rag" / dataset / "summary.json"
    exfiltration_control_summary_path = (
        results_dir / "attacks" / "exfiltration" / f"{dataset}_control" / "summary.json"
    )

    clean_summary = load_summary(validation_summary_path, required=True)
    instruction_summary = load_summary(instruction_summary_path, required=True)
    exfiltration_summary = load_summary(exfiltration_summary_path, required=True)
    poisoned_summary = load_summary(poisoned_summary_path, required=True)
    exfiltration_control_summary = load_summary(exfiltration_control_summary_path, required=False)

    output_dir = results_dir / "thesis" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    attack_rows = build_attack_effectiveness_rows(
        dataset,
        instruction_summary=instruction_summary,
        exfiltration_summary=exfiltration_summary,
        poisoned_summary=poisoned_summary,
        exfiltration_control_summary=exfiltration_control_summary,
    )
    clean_rows = build_clean_utility_rows(dataset, clean_summary=clean_summary)
    latency_rows = build_latency_overhead_rows(
        dataset,
        clean_summary=clean_summary,
        instruction_summary=instruction_summary,
        exfiltration_summary=exfiltration_summary,
        poisoned_summary=poisoned_summary,
    )
    compact_rows = build_compact_overview_rows(
        dataset,
        clean_summary=clean_summary,
        instruction_summary=instruction_summary,
        exfiltration_summary=exfiltration_summary,
        poisoned_summary=poisoned_summary,
    )

    attack_csv_path = output_dir / "attack_effectiveness_by_config.csv"
    clean_csv_path = output_dir / "clean_utility_false_positives_by_config.csv"
    latency_csv_path = output_dir / "latency_overhead_by_config.csv"
    compact_csv_path = output_dir / "cross_attack_compact_overview.csv"
    summary_json_path = output_dir / "summary.json"
    manifest_json_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"

    write_csv(
        attack_csv_path,
        attack_rows,
        fieldnames=list(attack_rows[0].keys()),
    )
    write_csv(
        clean_csv_path,
        clean_rows,
        fieldnames=list(clean_rows[0].keys()),
    )
    write_csv(
        latency_csv_path,
        latency_rows,
        fieldnames=list(latency_rows[0].keys()),
    )
    write_csv(
        compact_csv_path,
        compact_rows,
        fieldnames=list(compact_rows[0].keys()),
    )

    manifest = {
        "dataset": dataset,
        "inputs": {
            "clean_summary": str(validation_summary_path),
            "instruction_summary": str(instruction_summary_path),
            "exfiltration_summary": str(exfiltration_summary_path),
            "poisoned_summary": str(poisoned_summary_path),
            "exfiltration_control_summary": str(exfiltration_control_summary_path),
            "exfiltration_control_summary_exists": exfiltration_control_summary_path.exists(),
        },
        "outputs": {
            "attack_effectiveness_by_config_csv": str(attack_csv_path),
            "clean_utility_false_positives_by_config_csv": str(clean_csv_path),
            "latency_overhead_by_config_csv": str(latency_csv_path),
            "cross_attack_compact_overview_csv": str(compact_csv_path),
            "summary_json": str(summary_json_path),
            "readme_md": str(readme_path),
        },
    }
    summary_payload = {
        "dataset": dataset,
        "metric_definitions": metric_definitions(),
        "attack_effectiveness_rows": attack_rows,
        "clean_utility_rows": clean_rows,
        "latency_rows": latency_rows,
        "compact_overview_rows": compact_rows,
        "notes": {
            "legacy_logs_policy": "Legacy or incomplete log schemas are excluded from thesis-facing reporting.",
            "clean_metrics_role": "Clean metrics remain separate from attack metrics and are the basis for utility and benign false-positive reporting.",
            "poisoned_rag_denominator_note": "PoisonedRAG results use their own explicit denominator and must not be silently compared to n=20 attack sets without showing n.",
        },
    }

    write_json(summary_json_path, summary_payload)
    write_json(manifest_json_path, manifest)
    readme_path.write_text(build_readme_text(dataset, manifest), encoding="utf-8")

    print(f"Saved attack effectiveness CSV -> {attack_csv_path}")
    print(f"Saved clean utility CSV -> {clean_csv_path}")
    print(f"Saved latency overhead CSV -> {latency_csv_path}")
    print(f"Saved compact overview CSV -> {compact_csv_path}")
    print(f"Saved combined summary -> {summary_json_path}")
    print(f"Saved manifest -> {manifest_json_path}")
    print(f"Saved README -> {readme_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build thesis-ready cross-attack summaries.")
    parser.add_argument("--dataset", default=DEFAULT_CLOSE_ENDED_MAIN_DATASET)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    build_thesis_results(dataset=args.dataset, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
