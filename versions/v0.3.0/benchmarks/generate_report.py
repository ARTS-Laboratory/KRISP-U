"""Generate a compact Markdown summary from a benchmark output directory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def generate_report(output: Path) -> Path:
    rows = list(csv.DictReader((output / "final_metrics.csv").open(encoding="utf-8")))
    lines = [
        "# KRISP-U v0.3.0 benchmark report",
        "",
        "This report is diagnostic; it is not final performance evidence.",
        "",
        "| Field | Method | NRMSE | R² |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['field']} | {row['method']} | {float(row['nrmse']):.6g} | {float(row['r2']):.6g} |"
        )
    path = output / "benchmark_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(generate_report(args.output))


if __name__ == "__main__":
    main()
