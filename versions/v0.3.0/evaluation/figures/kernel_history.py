"""Kernel-event history figure built from serialized event records."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_kernel_history(rows: Iterable[dict[str, Any]], path: Path, *, dpi: int = 150) -> None:
    records = sorted(rows, key=lambda row: int(row["sample_count"]))
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    if not records:
        axes[0].set_title("Kernel history (no selection events)")
    else:
        counts = [int(row["sample_count"]) for row in records]
        families = [
            str(row.get("selected_family", row.get("selected_kernel_id", "unknown")))
            for row in records
        ]
        axes[0].step(counts, families, where="post")
        axes[0].set_ylabel("family")
        scales = [
            _parse_vector(row.get("length_scales", row.get("current_length_scales", "")))
            for row in records
        ]
        for dimension in range(max((len(value) for value in scales), default=0)):
            axes[1].plot(
                counts,
                [value[dimension] if len(value) > dimension else np.nan for value in scales],
                label=f"x{dimension + 1}",
            )
        axes[1].set_ylabel("ARD scale")
        for dimension in range(max((len(value) for value in scales), default=0)):
            lower = [
                _vector(row.get("length_scale_minimums"))[dimension]
                for row in records
                if len(_vector(row.get("length_scale_minimums"))) > dimension
            ]
            upper = [
                _vector(row.get("length_scale_maximums"))[dimension]
                for row in records
                if len(_vector(row.get("length_scale_maximums"))) > dimension
            ]
            if lower:
                axes[1].axhline(lower[-1], color=f"C{dimension}", linestyle=":", alpha=0.5)
            if upper:
                axes[1].axhline(upper[-1], color=f"C{dimension}", linestyle="--", alpha=0.5)
        axes[2].plot(
            counts,
            [_number(row.get("validation_score", row.get("selection_score"))) for row in records],
            label="current family",
        )
        axes[2].plot(
            counts,
            [
                _number(row.get("challenger_score", row.get("challenger_validation_score")))
                for row in records
            ],
            label="best challenger",
        )
        axes[2].set_ylabel("buffered score")
        reason_markers = {
            "bound-contact trigger": ("D", "tab:orange"),
            "score-degradation trigger": ("v", "tab:purple"),
            "maximum-interval trigger": ("s", "tab:green"),
            "fit-failure trigger": ("X", "tab:red"),
        }
        labels_seen: set[str] = set()
        for axis in axes:
            for row in records:
                if _truth(row.get("reselection_triggered")):
                    retained = not (
                        _truth(row.get("switch_accepted"))
                        and row.get("previous_family") not in (None, "", row.get("selected_family"))
                    )
                    label = "reselection evaluated, retained" if retained else "reselection accepted"
                    axis.axvline(
                        int(row["sample_count"]),
                        color="black" if retained else "crimson",
                        alpha=0.24,
                        linewidth=0.8,
                        label=label if label not in labels_seen else None,
                    )
                    labels_seen.add(label)
                    for reason in str(row.get("reselection_reasons", "")).split(";"):
                        marker = reason_markers.get(reason.strip())
                        if marker is None:
                            continue
                        marker_shape, color = marker
                        if axis is axes[2]:
                            y_value = _number(row.get("challenger_score"))
                        elif axis is axes[1]:
                            y_value = (_parse_vector(row.get("length_scales")) or [0.5])[0]
                        else:
                            y_value = 0.5
                        axis.scatter(
                            int(row["sample_count"]),
                            y_value,
                            marker=marker_shape,
                            color=color,
                            s=35,
                            zorder=6,
                            label=reason if reason not in labels_seen else None,
                        )
                        labels_seen.add(reason)
                if _truth(row.get("switch_accepted")) and row.get("previous_family") not in (
                    None,
                    "",
                    row.get("selected_family"),
                ):
                    axis.scatter(
                        int(row["sample_count"]),
                        0.5,
                        marker="*",
                        color="red",
                        zorder=5,
                        label="accepted family switch" if "accepted family switch" not in labels_seen else None,
                    )
                    labels_seen.add("accepted family switch")
        axes[1].legend(fontsize=7)
        axes[2].legend(fontsize=7)
        axes[0].set_title("Global kernel history | reselection events and accepted switches")
    axes[2].set_xlabel("sample count")
    for axis in axes:
        axis.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _parse_vector(value: Any) -> list[float]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [float(item) for item in value.replace(",", ";").split(";") if item]
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]


def _vector(value: Any) -> list[float]:
    return _parse_vector(value)


def _number(value: Any) -> float:
    if value in (None, ""):
        return np.nan
    return float(value)


def _truth(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() in {"true", "1", "yes"})


__all__ = ["plot_kernel_history"]
