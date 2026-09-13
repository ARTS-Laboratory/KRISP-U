"""Shared visual language for presentation figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

BACKGROUND = "#F7F8FA"
INK = "#17212B"
MUTED = "#5E6A75"
GRID = "#D8DEE5"
BLUE = "#146C94"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
GOLD = "#E9C46A"
PURPLE = "#6C63A8"
RED = "#C44536"

METHOD_COLORS = {
    "Random": MUTED,
    "LHS": GOLD,
    "Maxmin": PURPLE,
    "Maximin": PURPLE,
    "Latin hypercube": GOLD,
    "KRISP-U": ORANGE,
    "posterior std": BLUE,
}


def configure() -> None:
    """Apply consistent typography and axes defaults."""

    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.frameon": False,
            "figure.dpi": 120,
        }
    )


def style_axis(axis: Any, *, grid: bool = True) -> None:
    """Polish an axis without hiding the underlying quantitative structure."""

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    if grid:
        axis.grid(color=GRID, linewidth=0.7, alpha=0.65)
        axis.set_axisbelow(True)


def save_figure(figure: Figure, path: Path, *, dpi: int = 180) -> Path:
    """Save one figure and close it to keep batch generation memory bounded."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return path


def add_panel_label(axis: Any, label: str) -> None:
    """Add a compact panel label in a stable position."""

    axis.text(
        -0.12,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        color=INK,
        va="bottom",
    )


__all__ = [
    "BACKGROUND",
    "BLUE",
    "GOLD",
    "GRID",
    "INK",
    "METHOD_COLORS",
    "MUTED",
    "ORANGE",
    "PURPLE",
    "RED",
    "TEAL",
    "add_panel_label",
    "configure",
    "save_figure",
    "style_axis",
]
