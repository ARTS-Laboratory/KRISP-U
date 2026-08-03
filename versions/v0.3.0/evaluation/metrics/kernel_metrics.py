"""Kernel-study scalar aggregation over completed event records."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def kernel_auc_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute right-endpoint NRMSE AUC without refitting any model."""

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["field"]), int(row["trial"]), str(row["selection_mode"])), []
        ).append(row)
    result: list[dict[str, Any]] = []
    for (field, trial, mode), values in sorted(grouped.items()):
        values.sort(key=lambda item: int(item["sample_count"]))
        counts = np.asarray([int(item["sample_count"]) for item in values], dtype=float)
        nrmse = np.asarray([float(item["nrmse"]) for item in values], dtype=float)
        result.append(
            {
                "field": field,
                "trial": trial,
                "selection_mode": mode,
                "nrmse_auc": float(np.sum(nrmse[1:] * np.diff(counts))),
            }
        )
    return result


def aggregate_recovery_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate selected-vs-true kernel families from completed rows."""

    grouped: dict[tuple[str, str], int] = {}
    totals: dict[str, int] = {}
    for row in rows:
        true_family = str(row["true_field_family"])
        selected_family = str(row["selected_kernel_family"])
        grouped[(true_family, selected_family)] = grouped.get((true_family, selected_family), 0) + 1
        totals[true_family] = totals.get(true_family, 0) + 1
    return [
        {
            "true_field_family": true,
            "selected_kernel_family": selected,
            "selection_count": count,
            "selection_percentage": 100.0 * count / totals[true],
        }
        for (true, selected), count in sorted(grouped.items())
    ]


__all__ = ["aggregate_recovery_rows", "kernel_auc_rows"]
