"""Paired acquisition-method benchmark with a frozen kernel schedule."""

from __future__ import annotations

import csv
import platform
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.figures.field_plots import (
    plot_learning_curves,
    plot_paired_differences,
    plot_sampling_paths,
)
from evaluation.figures.sequential import save_sequential_visuals
from evaluation.metrics.reconstruction import nrmse_auc
from evaluation.runners.config import (
    jackknife_config,
    minimum_normalized_distance,
    prepare_suite_output,
)
from evaluation.runners.design import initial_design, make_field, regular_grid
from evaluation.runners.output import save_spatial_state
from evaluation.runners.sequential import run_sequential_design
from krispu.candidates import generate_candidates
from krispu.config import GPRConfig
from krispu.kernels.selection import KernelSelectionResult

ACQUISITION_METHODS = (
    "krispu",
    "raw_jackknife_sensitivity",
    "posterior_std",
    "random",
    "lhs",
    "maximin",
)
METHOD_ALIASES = {"krispu": "support_adjusted_krispu"}


def run_acquisition_comparison_study(config: dict[str, Any], output_root: Path) -> Path:
    """Run Study B using one selected/fitted kernel schedule per paired trial."""

    output = prepare_suite_output(output_root, config["experiment_name"])
    _make_output_dirs(output)
    (output / "config_resolved.yaml").write_text(
        __import__("yaml").safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    records: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    candidate_scores: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []

    for field_index, field_name in enumerate(config["fields"]):
        field_seed = int(config["base_seed"]) + field_index * 100_000 + 1
        field = make_field(field_name, field_seed)
        evaluation = regular_grid(field.domain, int(config["evaluation_grid_size"]))
        true_evaluation = field.evaluate(evaluation)
        final_states: dict[str, Any] = {}
        for trial in range(int(config["trials"])):
            trial_seed = int(config["base_seed"]) + field_index * 100_000 + trial * 10_000
            candidate_seed = trial_seed + 3
            initial_seed_list = config.get("initial_design_seeds")
            initial_seed = (
                int(initial_seed_list[trial % len(initial_seed_list)])
                if initial_seed_list
                else trial_seed + 2
            )
            candidate_pool = generate_candidates(
                field.domain, int(config["candidate_count"]), "lhs", candidate_seed
            )
            initial, jackknife_eligible = initial_design(
                config["initial_design"],
                field.domain,
                int(config["initial_sample_count"]),
                float(config["initial_boundary_margin"]),
                initial_seed,
                return_eligibility=True,
            )
            schedule_states = run_sequential_design(
                field.evaluate,
                field.domain,
                initial,
                candidate_pool,
                evaluation,
                "support_adjusted_krispu",
                int(config["final_budget"]),
                trial_seed + 90,
                field_name=field_name,
                trial=trial,
                true_evaluation=true_evaluation,
                gpr_config=GPRConfig(random_state=trial_seed + 90, jackknife=jackknife_config(config)),
                initial_jackknife_eligible=jackknife_eligible,
                minimum_normalized_distance=minimum_normalized_distance(config),
                boundary_margin=float(config["initial_boundary_margin"]),
                kernel_selection_config=_kernel_schedule_config(field, config),
                selection_mode_label="hybrid_correct_profile",
            )
            schedule = _schedule_from_states(schedule_states)
            for state in schedule_states:
                result = state.kernel_selection_result
                if isinstance(result, KernelSelectionResult):
                    history.append(
                        {
                            "field": field_name,
                            "trial": trial,
                            **result.selected_record(),
                        }
                    )
                    for score in result.candidate_scores:
                        row = score.as_record(
                            state.sample_count,
                            result.previous_kernel_id,
                            result.selected_kernel_id,
                            result.switch_accepted,
                        )
                        candidate_scores.append({"field": field_name, "trial": trial, **row})
            method_states: dict[str, list[Any]] = {}
            for method_index, display_method in enumerate(ACQUISITION_METHODS):
                method = METHOD_ALIASES.get(display_method, display_method)
                states = run_sequential_design(
                    field.evaluate,
                    field.domain,
                    initial,
                    candidate_pool,
                    evaluation,
                    method,
                    int(config["final_budget"]),
                    trial_seed + 100 + method_index,
                    field_name=field_name,
                    trial=trial,
                    true_evaluation=true_evaluation,
                    gpr_config=GPRConfig(random_state=trial_seed + 100 + method_index, jackknife=jackknife_config(config)),
                    initial_jackknife_eligible=jackknife_eligible,
                    minimum_normalized_distance=minimum_normalized_distance(config),
                    boundary_margin=float(config["initial_boundary_margin"]),
                    kernel_schedule=schedule,
                    selection_mode_label="hybrid_correct_profile",
                )
                method_states[display_method] = states
                final_states[display_method] = states[-1]
                for state in states:
                    row = state.scalar_record
                    row.update(
                        {
                            "method": display_method,
                            "mode": "hybrid_correct_profile",
                            "study": "B",
                        }
                    )
                    records.append(row)
                    if state.sample_count == int(config["final_budget"]):
                        finals.append(row)
                    save_spatial_state(
                        output
                        / "spatial_arrays"
                        / (
                            f"{field_name}_{display_method}_hybrid_correct_profile_"
                            f"trial{trial}_n{state.sample_count}.npz"
                        ),
                        state,
                    )
                if _visuals_enabled(config):
                    save_sequential_visuals(
                        states,
                        output,
                        save_gif=bool(config.get("save_gifs", True)),
                        save_snapshots=bool(config.get("save_png_snapshots", True)),
                        snapshot_every=config.get("snapshot_every"),
                        snapshot_sample_counts=config.get("snapshot_sample_counts"),
                        frame_duration_ms=int(config.get("frame_duration_ms", 500)),
                        final_frame_duration_ms=int(config.get("final_frame_duration_ms", 1500)),
                        dpi=int(config.get("dpi", 180)),
                        annotate_point_order=bool(config.get("annotate_point_order", True)),
                        save_point_layout_gif=True,
                        save_contact_sheet=bool(config.get("save_contact_sheet", True)),
                    )
            krispu_final = method_states["krispu"][-1]
            for baseline in ACQUISITION_METHODS[1:]:
                base_final = method_states[baseline][-1]
                paired.append(
                    {
                        "field": field_name,
                        "trial": trial,
                        "baseline": baseline,
                        "delta_nrmse": krispu_final.metrics.nrmse - base_final.metrics.nrmse,
                        "delta_auc": _trial_auc(method_states["krispu"])
                        - _trial_auc(method_states[baseline]),
                        "krispu_nrmse": krispu_final.metrics.nrmse,
                        "baseline_nrmse": base_final.metrics.nrmse,
                    }
                )
            for method, states in method_states.items():
                auc_rows.append(
                    {
                        "field": field_name,
                        "method": method,
                        "mode": "hybrid_correct_profile",
                        "trial": trial,
                        "nrmse_auc": _trial_auc(states),
                        "final_nrmse": states[-1].metrics.nrmse,
                    }
                )
            seeds.append(
                {
                    "field": field_name,
                    "trial": trial,
                    "field_seed": field_seed,
                    "initial_design_seed": initial_seed,
                    "candidate_seed": candidate_seed,
                    "kernel_selection_seed": trial_seed + 90,
                    "method_seeds": {
                        method: trial_seed + 100 + index
                        for index, method in enumerate(ACQUISITION_METHODS)
                    },
                }
            )
        plot_sampling_paths(
            field,
            final_states,
            output / "snapshots" / "sampling_paths" / f"{field_name}_sampling_paths.png",
            bool(config.get("save_pdf", False)),
        )

    plot_learning_curves(
        records,
        output / "figures" / "learning_curves",
        int(config["trials"]) > 1,
    )
    plot_paired_differences(
        paired, output / "figures" / "paired_performance", bool(config.get("save_pdf", False))
    )
    _write_csv(output / "nrmse_auc.csv", auc_rows)
    _write_acquisition_figures(output, records, paired, history)
    _write_csv(output / "iteration_metrics.csv", records)
    _write_csv(output / "final_metrics.csv", finals)
    _write_csv(output / "paired_comparisons.csv", paired)
    _write_csv(output / "kernel_selection_history.csv", history)
    _write_csv(
        output / "kernel_hyperparameter_history.csv",
        [
            {
                "field": row.get("field"),
                "trial": row.get("trial"),
                "sample_count": row.get("sample_count"),
                "selection_mode": row.get("selection_mode"),
                "selected_kernel_id": row.get("selected_kernel_id"),
                "optimized_hyperparameters": row.get("optimized_hyperparameters"),
            }
            for row in history
        ],
    )
    _write_csv(output / "kernel_candidate_scores.csv", candidate_scores)
    _write_manifest(output, config, seeds)
    _write_report(output, records, finals, paired, auc_rows, history)
    return output


def _kernel_schedule_config(field: Any, config: dict[str, Any]) -> dict[str, Any]:
    family = str(field.metadata.get("field_family", "smooth_global"))
    profile = {
        "smooth_global": "smooth_global",
        "rough_single_scale": "rough_single_scale",
        "rough_multiscale": "rough_multiscale",
        "trend_plus_local": "trend_plus_local",
        "periodic": "periodic",
    }.get(family, "smooth_global")
    selection = {
        key: value for key, value in config.get("kernel_selection", {}).items() if key != "enabled"
    }
    return {
        "mode": "hybrid",
        "profile": profile,
        **selection,
    }


def _schedule_from_states(states: list[Any]) -> dict[int, tuple[str, Any]]:
    schedule: dict[int, tuple[str, Any]] = {}
    for state in states:
        result = state.kernel_selection_result
        if not isinstance(result, KernelSelectionResult):
            raise TypeError("Kernel schedule source did not record a selection result.")
        schedule[state.sample_count] = (result.selected_kernel_id, result.fitted_kernel)
    return schedule


def _trial_auc(states: list[Any]) -> float:
    return nrmse_auc(
        [state.sample_count for state in states],
        [state.metrics.nrmse for state in states],
    )


def _make_output_dirs(output: Path) -> None:
    for name in (
        "spatial_arrays",
        "animations/process",
        "animations/point_progress",
        "snapshots/frames",
        "snapshots/contact_sheets",
        "snapshots/acquisition_decomposition",
        "snapshots/sampling_paths",
        "figures/learning_curves",
        "figures/kernel_selection",
        "figures/kernel_hyperparameters",
        "figures/acquisition_comparison",
        "figures/uncertainty_error",
        "figures/paired_performance",
        "reports",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)


def _visuals_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("save_gifs", False) or config.get("save_png_snapshots", False))


def _write_acquisition_figures(
    output: Path,
    records: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> None:
    fields = sorted({str(row["field"]) for row in records})
    for field in fields:
        subset = [row for row in records if row["field"] == field]
        for metric, title in (
            ("nrmse", "NRMSE"),
            ("r2", "R²"),
            ("p95_absolute_error", "p95 absolute error"),
            ("max_absolute_error", "maximum absolute error"),
        ):
            figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
            for method in ACQUISITION_METHODS:
                values = sorted(
                    [row for row in subset if row["method"] == method],
                    key=lambda row: int(row["sample_count"]),
                )
                if values:
                    axis.plot(
                        [row["sample_count"] for row in values],
                        [float(row[metric]) for row in values],
                        marker="o",
                        label=method,
                    )
            axis.set(title=f"{title} versus measurements | {field}", xlabel="measurements")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            _save(figure, output / "figures" / "acquisition_comparison" / f"{field}_{metric}.png")
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for method in ACQUISITION_METHODS:
        values = [
            float(row["nrmse_auc"])
            for row in _read_rows(output / "nrmse_auc.csv")
            if row["method"] == method
        ]
        if values:
            axis.scatter([method] * len(values), values)
    axis.set(title="NRMSE AUC by acquisition method", ylabel="NRMSE AUC")
    axis.tick_params(axis="x", rotation=35)
    _save(figure, output / "figures" / "acquisition_comparison" / "nrmse_auc.png")
    if history:
        figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for field in fields:
            values = [row for row in history if row["field"] == field]
            axis.step(
                [int(row["sample_count"]) for row in values],
                [str(row["selected_kernel_id"]) for row in values],
                where="post",
                label=field,
            )
        axis.set(title="Frozen kernel schedule used by Study B", xlabel="measurements")
        axis.legend(fontsize=8)
        _save(figure, output / "figures" / "kernel_selection" / "kernel_schedule.png")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["sample_count"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(output: Path, config: dict[str, Any], seeds: list[dict[str, Any]]) -> None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    dependencies: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "sklearn", "matplotlib", "yaml", "PIL"):
        try:
            module = __import__(name)
            dependencies[name] = getattr(module, "__version__", None)
        except ImportError:
            dependencies[name] = None
    manifest = {
        "git_commit": commit,
        "package_version": "0.3.0",
        "python_version": platform.python_version(),
        "dependency_versions": dependencies,
        "configuration": config,
        "seeds": seeds,
        "kernel_schedule": "hybrid_correct_profile schedule frozen for every acquisition method",
        "metric_definitions": {
            "nrmse_auc": "sum of right-endpoint NRMSE times sample-count increments",
            "krispu_uncertainty": "buffered-jackknife field sensitivity times sqrt(kernel support deficit)",
        },
    }
    (output / "manifest.yaml").write_text(
        __import__("yaml").safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )


def _write_report(
    output: Path,
    records: list[dict[str, Any]],
    finals: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    auc_rows: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> None:
    lines = [
        "# KRISP-U v0.3.0 acquisition comparison report",
        "",
        "Study B uses one hybrid-correct fitted-kernel schedule per paired trial.",
        "",
        "## Major artifacts",
        "",
        "- Process GIFs: `animations/process/`",
        "- Snapshot frames and contact sheets: `snapshots/`",
        "- Learning curves: `figures/learning_curves/` and `figures/acquisition_comparison/`",
        "- Kernel schedule: `figures/kernel_selection/kernel_schedule.png`",
        "",
        "## Final NRMSE and NRMSE AUC",
        "",
        "| Field | Method | Final NRMSE | NRMSE AUC |",
        "|---|---|---:|---:|",
    ]
    for row in sorted(auc_rows, key=lambda item: (str(item["field"]), float(item["nrmse_auc"]))):
        lines.append(
            f"| {row['field']} | {row['method']} | {float(row['final_nrmse']):.6g} | "
            f"{float(row['nrmse_auc']):.6g} |"
        )
    lines.extend(
        (
            "",
            "## Paired KRISP-U differences",
            "",
            "| Field | Baseline | ΔNRMSE | ΔAUC |",
            "|---|---|---:|---:|",
        )
    )
    for row in paired:
        lines.append(
            f"| {row['field']} | {row['baseline']} | {float(row['delta_nrmse']):.6g} | "
            f"{float(row['delta_auc']):.6g} |"
        )
    lines.extend(
        (
            "",
            "## Limitations",
            "",
            "The spatial diagnostics measure ranking quality, not probabilistic calibration. Rough fields may legitimately produce dense sampling.",
            "",
            "Out-of-scope files changed: none",
        )
    )
    (output / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
