"""Generate a compact Markdown summary from a benchmark output directory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


def generate_report(output: Path) -> Path:
    """Write the report and return its path."""

    rows = list(csv.DictReader((output / "final_metrics.csv").open(encoding="utf-8")))
    summary_path = output / "audit_summary.csv"
    summaries = (
        list(csv.DictReader(summary_path.open(encoding="utf-8"))) if summary_path.exists() else []
    )
    lines = [
        "# KRISP-U v0.3.0 benchmark report",
        "",
        "This report is diagnostic; it is not final performance evidence.",
        "",
        f"Animations: `{_file_summary(output / 'animations')}`",
        f"Comparison figures: `{_file_summary(output / 'comparisons')}`",
        "",
        "## Final metrics",
        "",
        "| Field | Method | Trial | NRMSE | R² |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['field']} | {row['method']} | {row['trial']} | "
            f"{float(row['nrmse']):.6g} | {float(row['r2']):.6g} |"
        )

    lines.extend(("", "## Method ranking by mean final NRMSE", ""))
    for field in sorted({row["field"] for row in rows}):
        means: dict[str, list[float]] = {}
        for row in rows:
            if row["field"] == field:
                means.setdefault(row["method"], []).append(float(row["nrmse"]))
        ranking = sorted((float(np.mean(values)), method) for method, values in means.items())
        lines.append(
            f"- `{field}`: " + ", ".join(f"{method} ({value:.4g})" for value, method in ranking)
        )

    if summaries:
        lines.extend(
            (
                "",
                "## Required selection diagnostics",
                "",
                "| Field | Method | Trial | Final NRMSE | NRMSE AUC | Median nearest distance | Fraction within 0.05 | Fraction correlation > 0.95 |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            )
        )
        for row in summaries:
            lines.append(
                f"| {row['field']} | {row['method']} | {row['trial']} | "
                f"{float(row['final_nrmse']):.6g} | {float(row['nrmse_auc']):.6g} | "
                f"{_optional_float(row.get('median_nearest_observation_distance'))} | "
                f"{_optional_float(row.get('fraction_selections_within_0_05_normalized_distance'))} | "
                f"{_optional_float(row.get('fraction_selections_kernel_correlation_above_0_95'))} |"
            )

    noisy = [row for row in summaries if row["field"] == "noisy_baseline"]
    if noisy:
        lines.extend(("", "## Noisy-baseline field", ""))
        lines.append("| Method | Final NRMSE | NRMSE AUC | Final R² |")
        lines.append("|---|---:|---:|---:|")
        methods = sorted({row["method"] for row in noisy})
        for method in methods:
            method_rows = [row for row in noisy if row["method"] == method]
            lines.append(
                f"| {method} | {_mean(method_rows, 'final_nrmse'):.6g} | "
                f"{_mean(method_rows, 'nrmse_auc'):.6g} | {_mean(method_rows, 'final_r2'):.6g} |"
            )
        krispu = _mean(
            [row for row in noisy if row["method"] == "support_adjusted_krispu"],
            "final_nrmse",
        )
        baseline_values = [
            _mean([row for row in noisy if row["method"] == method], "final_nrmse")
            for method in methods
            if method != "support_adjusted_krispu"
        ]
        if baseline_values:
            best_baseline = min(baseline_values)
            lines.extend(
                (
                    "",
                    (
                        "KRISP-U outperformed the best baseline on noisy_baseline by final "
                        f"NRMSE: {'yes' if krispu < best_baseline else 'no'} "
                        f"(KRISP-U={krispu:.6g}, best baseline={best_baseline:.6g})."
                    ),
                )
            )

    path = output / "reports" / "benchmark_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _optional_float(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.6g}"


def _file_summary(directory: Path) -> str:
    count = sum(1 for path in directory.rglob("*") if path.is_file())
    return f"{directory.name}/ ({count} files)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(generate_report(args.output))


if __name__ == "__main__":
    main()
