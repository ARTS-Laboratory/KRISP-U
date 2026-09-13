"""KRISP-U workflow schematic for a presentation slide."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .styles import BLUE, GOLD, INK, MUTED, ORANGE, PURPLE, RED, TEAL, save_figure


def generate_workflow_figure(output_dir: Path, *, dpi: int = 180) -> Path:
    """Write one compact end-to-end KRISP-U workflow schematic."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(14, 7.3), constrained_layout=True)
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 7.3)
    axis.axis("off")
    axis.text(
        0.2,
        6.85,
        "KRISP-U workflow",
        fontsize=20,
        fontweight="bold",
        color=INK,
        va="top",
    )
    axis.text(
        0.2,
        6.36,
        "Candidate-level uncertainty combines what the data say with where the fitted kernel has support.",
        fontsize=11,
        color=MUTED,
        va="top",
    )

    nodes = [
        (0.35, 3.55, 2.15, 1.48, "1", "Candidate space", "unmeasured x", BLUE),
        (3.00, 3.55, 2.15, 1.48, "2", "Measure response", "y(xᵢ)", TEAL),
        (5.65, 3.55, 2.55, 1.48, "3", "Fit response-standardized GP", "one global ARD kernel", PURPLE),
        (8.75, 3.55, 2.15, 1.48, "4", "Leave one out", "field sensitivity S(x)", GOLD),
        (11.40, 3.55, 2.15, 1.48, "5", "Rank candidates", "U(x) = S(x)√D(x)", ORANGE),
    ]
    for node in nodes:
        _box(axis, *node)
    for left, right in pairwise(nodes):
        _arrow(axis, left[0] + left[2], left[1] + left[3] / 2, right[0], right[1] + right[3] / 2)

    _box(
        axis,
        5.45,
        1.10,
        3.10,
        1.38,
        "A",
        "Support adjustment",
        "D(x) = 1 − maxᵢ corrₖ(x, xᵢ)",
        ORANGE,
    )
    _arrow(axis, 9.8, 3.55, 8.0, 2.48, color=ORANGE, connectionstyle="arc3,rad=-0.2")
    _arrow(axis, 8.0, 2.48, 8.0, 3.55, color=ORANGE, connectionstyle="arc3,rad=0.2")

    _box(
        axis,
        9.55,
        1.10,
        3.10,
        1.38,
        "B",
        "Acquire next point",
        "xₙ₊₁ = arg maxₓ U(x)",
        RED,
    )
    _arrow(axis, 12.45, 3.55, 11.1, 2.48, color=RED, connectionstyle="arc3,rad=-0.22")
    _arrow(axis, 11.1, 2.48, 4.1, 3.55, color=RED, connectionstyle="arc3,rad=0.20")

    axis.text(
        0.35,
        0.52,
        "Repeat until the response field is reconstructed at the required accuracy.",
        color=MUTED,
        fontsize=11,
    )
    return save_figure(figure, output_dir / "krispu_workflow.png", dpi=dpi)


def _box(
    axis: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.12",
        facecolor="white",
        edgecolor=color,
        linewidth=2.0,
    )
    axis.add_patch(patch)
    axis.text(x + 0.18, y + height - 0.25, label, fontsize=10, fontweight="bold", color=color)
    axis.text(
        x + width / 2,
        y + height * 0.58,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=INK,
        wrap=True,
    )
    axis.text(
        x + width / 2,
        y + 0.28,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
    )


def _arrow(
    axis: Any,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    *,
    color: str = MUTED,
    connectionstyle: str = "arc3",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            (x_start, y_start),
            (x_end, y_end),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.5,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


__all__ = ["generate_workflow_figure"]
