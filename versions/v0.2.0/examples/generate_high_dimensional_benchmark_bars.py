"""Generate high-dimensional KRISP-U baseline-comparison bar charts."""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.exceptions import ConvergenceWarning

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from krispu import BenchmarkResult, run_benchmark  # noqa: E402

DATASETS = ("hartmann_3d", "additive_5d", "hartmann_6d")
METHODS = ("krispu", "random", "grid", "lhs")


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    output_dir = Path("benchmark_outputs/high_dimensional")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float | str]] = []
    comparisons: list[dict[str, float | str]] = []

    for dataset_name in DATASETS:
        result = run_benchmark(
            dataset_name,
            methods=METHODS,
            budget=_budget_for(dataset_name),
            n_initial=_initial_for(dataset_name),
            n_trials=4,
            random_state=301,
            n_candidates=768,
            acquisition="uncertainty",
            optimize_continuous_acquisition=False,
            score_learning_curve=False,
            adaptive_kernel=False,
        )
        _save_raw_traces(result, output_dir)
        for method, values in result.summary(n_bootstrap=500).items():
            summaries.append({"dataset": dataset_name, "method": method, **values})

        for baseline in ("random", "grid", "lhs"):
            comparisons.append(
                {
                    "dataset": dataset_name,
                    "baseline": baseline,
                    "metric": "field_nrmse",
                    **result.compare_to_baseline(
                        "krispu",
                        baseline,
                        metric="field_nrmse",
                        n_bootstrap=500,
                    ),
                }
            )

    _write_csv(output_dir / "high_dimensional_summary.csv", summaries)
    _write_csv(output_dir / "high_dimensional_comparisons.csv", comparisons)
    _plot_grouped_bars(
        summaries,
        output_dir / "high_dimensional_field_nrmse_bars.png",
    )
    _plot_grouped_bars(
        summaries,
        output_dir / "high_dimensional_field_nmae_bars.png",
        metric="field_nmae_mean",
        ylabel="mean final field NMAE",
        title="High-dimensional field reconstruction error, mean absolute",
    )
    print(f"Wrote high-dimensional benchmark outputs to {output_dir}")


def _budget_for(dataset_name: str) -> int:
    if dataset_name == "hartmann_6d":
        return 31
    if dataset_name == "additive_5d":
        return 29
    return 24


def _initial_for(dataset_name: str) -> int:
    if dataset_name == "hartmann_6d":
        return 13
    if dataset_name == "additive_5d":
        return 11
    return 7


def _plot_grouped_bars(
    summaries: list[dict[str, float | str]],
    output_path: Path,
    metric: str = "field_nrmse_mean",
    ylabel: str = "mean final field NRMSE",
    title: str = "High-dimensional field reconstruction error",
) -> None:
    rows = [
        row
        for row in summaries
        if row["method"] in METHODS and np.isfinite(float(row[metric]))
    ]
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    x_axis = np.arange(len(datasets))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    for index, method in enumerate(METHODS):
        means = []
        lows = []
        highs = []
        for dataset in datasets:
            row = next(
                item
                for item in rows
                if item["dataset"] == dataset and item["method"] == method
            )
            mean = float(row[metric])
            means.append(mean)
            if metric == "field_nrmse_mean":
                lows.append(max(mean - float(row["field_nrmse_ci_low"]), 0.0))
                highs.append(max(float(row["field_nrmse_ci_high"]) - mean, 0.0))
            else:
                lows.append(0.0)
                highs.append(0.0)
        positions = x_axis + (index - 1.5) * width
        ax.bar(
            positions,
            means,
            width=width,
            label=method,
            yerr=np.asarray([lows, highs]) if any(highs) else None,
            capsize=3,
        )

    ax.set_xticks(x_axis)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_raw_traces(result: BenchmarkResult, output_dir: Path) -> None:
    payload = {
        "dataset": result.dataset_name,
        "budget": result.budget,
        "n_initial": result.n_initial,
        "methods": {
            method: [
                {
                    "seed": trace.seed,
                    "field_nrmse": trace.field_nrmse,
                    "field_nmae": trace.field_nmae,
                    "field_rmse": trace.field_rmse,
                    "field_mae": trace.field_mae,
                    "best_y": trace.best_y,
                }
                for trace in traces
            ]
            for method, traces in result.methods.items()
        },
    }
    path = output_dir / f"{result.dataset_name}_high_dimensional_raw_traces.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
