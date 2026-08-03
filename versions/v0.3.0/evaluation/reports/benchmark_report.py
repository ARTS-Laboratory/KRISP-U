"""Human-readable report generated from metric and kernel-event records."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def generate_report(output: Path) -> Path:
    rows = _read(output / "metrics" / "final.csv")
    aggregate = _read(output / "metrics" / "aggregate.csv")
    events = _read(output / "kernel" / "events.csv")
    lines = [
        "# KRISP-U benchmark report",
        "",
        "The benchmark separates sampled-GP covariance recovery from deterministic reconstruction performance.",
        "",
    ]
    if rows:
        lines.extend(
            [
                "## Field summaries",
                "",
                "| Field | Best final NRMSE | Best NRMSE AUC | Kernel sequence | Reselection n | Switch n | Final ARD scales | Near-neighbor rate |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for field in sorted({row["field"] for row in rows}):
            field_rows = [row for row in rows if row["field"] == field]
            final_by_method = _mean_by(field_rows, "method", "nrmse")
            auc_by_method = _mean_by(
                [row for row in aggregate if row["field"] == field], "method", "nrmse_auc"
            )
            event_rows = [row for row in events if row["field"] == field]
            sequence = _sequence(event_rows)
            reselection = (
                ", ".join(
                    sorted(
                        {
                            row["sample_count"]
                            for row in event_rows
                            if row.get("reselection_triggered", "").lower() == "true"
                        }
                    )
                )
                or "none"
            )
            switches = (
                ", ".join(
                    sorted(
                        {
                            row["sample_count"]
                            for row in event_rows
                            if row.get("switch_accepted", "").lower() == "true"
                            and row.get("previous_family")
                            not in (None, "", row.get("selected_family"))
                        }
                    )
                )
                or "none"
            )
            final_event = event_rows[-1] if event_rows else {}
            near = _mean(
                [row for row in aggregate if row["field"] == field],
                "fraction_selections_within_0_05_normalized_distance",
            )
            lines.append(
                f"| {field} | {_best(final_by_method)} | {_best(auc_by_method)} | {sequence} | {reselection} | {switches} | {final_event.get('length_scales', 'n/a')} | {near} |"
            )
    lines.extend(["", "## Kernel-event narrative", ""])
    for field in sorted({row["field"] for row in events}):
        lines.append(f"### {field}")
        for row in sorted(
            (item for item in events if item["field"] == field),
            key=lambda item: int(item["sample_count"]),
        ):
            reasons = row.get("reselection_reasons") or "routine optimization"
            if row.get("reselection_triggered", "").lower() == "true":
                previous = row.get("previous_family")
                selected = row.get("selected_family")
                switched = (
                    row.get("switch_accepted", "").lower() == "true"
                    and previous not in (None, "", selected)
                )
                if previous in (None, ""):
                    lines.append(
                        f"n={row['sample_count']}: initial family evaluation selected "
                        f"{selected}."
                    )
                elif switched:
                    lines.append(
                        f"n={row['sample_count']}: reselection triggered by {reasons}; "
                        f"switched {previous} -> {selected}."
                    )
                else:
                    lines.append(
                        f"n={row['sample_count']}: reselection triggered by {reasons}; "
                        "family retained."
                    )
        lines.append("")
    lines.extend(
        [
            "## Scientific cautions",
            "",
            "Deterministic development and canonical fields are response functions; this report does not assign them a true kernel. Synthetic recovery claims are restricted to fields whose metadata records an actual GP draw.",
            "",
            "Major failure observations are summarized from scalar metrics below; summary mode intentionally stores no per-step spatial arrays.",
            "",
        ]
    )
    lines.extend(["## Major failure observations", ""])
    for field in sorted({row["field"] for row in rows}):
        field_rows = [row for row in rows if row["field"] == field]
        observations = _failure_observations(field_rows)
        lines.append(f"- **{field}:** {observations}")
    lines.append("")
    path = output / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_by(rows: list[dict[str, str]], group: str, value: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get(value) not in (None, ""):
            grouped.setdefault(row[group], []).append(float(row[value]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _mean(rows: list[dict[str, str]], key: str) -> str:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return "n/a" if not values else f"{float(np.mean(values)):.4g}"


def _best(values: dict[str, float]) -> str:
    if not values:
        return "n/a"
    return min(values.items(), key=lambda item: item[1])[0]


def _sequence(rows: list[dict[str, str]]) -> str:
    values = []
    for row in sorted(rows, key=lambda item: int(item["sample_count"])):
        family = row.get("selected_family")
        if family and (not values or values[-1] != family):
            values.append(family)
    return " -> ".join(values) or "n/a"


def _failure_observations(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "no completed measurements"
    final = [row for row in rows if row.get("sample_count") == _maximum_count(rows)]
    nrmse = [float(row["nrmse"]) for row in final if row.get("nrmse") not in (None, "")]
    p95 = [
        float(row["p95_absolute_error"])
        for row in final
        if row.get("p95_absolute_error") not in (None, "")
    ]
    near = [
        float(row["near_neighbor_acquisition_rate"])
        for row in rows
        if row.get("near_neighbor_acquisition_rate") not in (None, "")
    ]
    notes: list[str] = []
    if nrmse and max(nrmse) > 0.25:
        notes.append("final NRMSE remains high")
    if p95 and max(p95) > 0.25:
        notes.append("large high-error tail")
    if near and float(np.mean(near)) > 0.25:
        notes.append("frequent near-neighbor acquisition")
    return "; ".join(notes) if notes else "no dominant scalar failure signal"


def _maximum_count(rows: list[dict[str, str]]) -> str:
    return str(max(int(row["sample_count"]) for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(generate_report(args.output))


if __name__ == "__main__":
    main()
