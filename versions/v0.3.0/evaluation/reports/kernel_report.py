"""Kernel-study report generation from completed tables."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def write_kernel_report(
    output: Path,
    final_rows: Iterable[dict[str, Any]],
    history_rows: Iterable[dict[str, Any]],
    experiment_name: str,
) -> Path:
    """Write a compact kernel report without accessing models or fields."""

    finals = list(final_rows)
    history = list(history_rows)
    lines = [
        "# KRISP-U v0.3.0 kernel-selection report",
        "",
        f"Configuration: `{experiment_name}`.",
        "",
        "| Field | Selection mode | Final NRMSE |",
        "|---|---|---:|",
    ]
    keys = sorted({(str(row["field"]), str(row["selection_mode"])) for row in finals})
    for field, mode in keys:
        values = [
            float(row["nrmse"])
            for row in finals
            if str(row["field"]) == field and str(row["selection_mode"]) == mode
        ]
        lines.append(f"| {field} | {mode} | {float(np.mean(values)):.6g} |")
    lines.extend(("", f"Recorded kernel events: {len(history)}.", ""))
    path = output / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = ["write_kernel_report"]
