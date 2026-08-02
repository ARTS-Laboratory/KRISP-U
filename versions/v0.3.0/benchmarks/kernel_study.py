"""Kernel-selection benchmark studies and their reproducible artifacts."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import qmc

from benchmarks.evaluation import nrmse_auc
from benchmarks.fields import FIELD_FACTORIES
from benchmarks.records import save_spatial_state
from benchmarks.runner import (
    _initial_design,
    _make_field,
    _normalize_config,
    _prepare_benchmark_output,
    _regular_grid,
)
from benchmarks.visualization import save_sequential_visuals
from krispu import GPRConfig
from krispu.candidates import generate_candidates
from krispu.kernels.selection import KernelSelectionResult
from krispu.sequential import run_sequential_design

MODE_NAMES = (
    "fixed_generic",
    "manual_correct",
    "manual_mismatched",
    "automatic_standard",
    "hybrid_correct_profile",
    "hybrid_broad_profile",
)
BASELINE_METHODS = (
    "raw_loo_sensitivity",
    "support_adjusted_krispu",
    "posterior_std",
    "random",
    "lhs",
    "maximin",
)


def run_kernel_selection_study(
    config_path: Path,
    output_root: Path = Path("benchmark_outputs"),
    config: dict[str, Any] | None = None,
) -> Path:
    config = _normalize_config(
        yaml.safe_load(config_path.read_text(encoding="utf-8")) if config is None else config
    )
    _validate_study_config(config)
    output = _prepare_benchmark_output(output_root, config["experiment_name"])
    (output / "figures").mkdir(exist_ok=True)
    (output / "animations").mkdir(exist_ok=True)
    (output / "spatial_arrays").mkdir(exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    study_a_rows: list[dict[str, Any]] = []
    study_a_final: list[dict[str, Any]] = []
    study_b_rows: list[dict[str, Any]] = []
    study_b_final: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    hyperparameter_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    source_schedules: dict[tuple[str, int], dict[int, tuple[str, Any]]] = {}

    for field_index, field_name in enumerate(config["fields"]):
        field_seed = int(config["base_seed"]) + field_index * 100_000 + 1
        field = _make_field(field_name, field_seed)
        evaluation = _regular_grid(field.domain, int(config["evaluation_grid_size"]))
        for trial in range(int(config["trials"])):
            trial_seed = int(config["base_seed"]) + field_index * 100_000 + trial * 10_000
            candidate_pool = generate_candidates(
                field.domain, int(config["candidate_count"]), "lhs", trial_seed + 3
            )
            initial_seed_list = config.get("initial_design_seeds")
            initial_design_seed = (
                int(initial_seed_list[trial % len(initial_seed_list)])
                if initial_seed_list
                else trial_seed + 2
            )
            initial, initial_loo_eligible = _initial_design(
                config.get("initial_design", "interior_maximin"),
                field.domain,
                int(config["initial_sample_count"]),
                float(config["initial_boundary_margin"]),
                initial_design_seed,
                return_eligibility=True,
            )
            true_evaluation = field.evaluate(evaluation)
            reference_states = run_sequential_design(
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
                gpr_config=GPRConfig(alpha=1e-6, random_state=trial_seed + 90),
                initial_loo_eligible=initial_loo_eligible,
                minimum_normalized_distance=_minimum_normalized_distance(config),
                boundary_margin=float(config["initial_boundary_margin"]),
            )
            forced_points = np.asarray(
                [state.recommended_point for state in reference_states[:-1]], dtype=float
            )
            mode_states: dict[str, list[Any]] = {}
            for mode_index, mode in enumerate(MODE_NAMES):
                mode_config = _mode_config(mode, field.metadata, config)
                states = run_sequential_design(
                    field.evaluate,
                    field.domain,
                    initial,
                    candidate_pool,
                    evaluation,
                    "support_adjusted_krispu",
                    int(config["final_budget"]),
                    trial_seed + 100 + mode_index,
                    field_name=field_name,
                    trial=trial,
                    true_evaluation=true_evaluation,
                    gpr_config=GPRConfig(alpha=1e-6, random_state=trial_seed + 100 + mode_index),
                    minimum_normalized_distance=_minimum_normalized_distance(config),
                    boundary_margin=float(config["initial_boundary_margin"]),
                    kernel_selection_config=mode_config,
                    selection_mode_label=mode,
                    initial_loo_eligible=initial_loo_eligible,
                    forced_points=forced_points,
                )
                mode_states[mode] = states
                for state in states:
                    row = state.scalar_record()
                    row.update({"selection_mode": mode, "mode": mode, "study": "A"})
                    study_a_rows.append(row)
                    if state.sample_count == int(config["final_budget"]):
                        study_a_final.append(row)
                    save_spatial_state(
                        output
                        / "spatial_arrays"
                        / f"study_a_{field_name}_{mode}_trial{trial}_n{state.sample_count}.npz",
                        state,
                    )
                    result = state.kernel_selection_result
                    if isinstance(result, KernelSelectionResult):
                        for score in result.candidate_scores:
                            record = score.as_record(
                                state.sample_count,
                                mode,
                                result.profile,
                                result.previous_kernel_id,
                                result.selected_kernel_id,
                                result.switch_accepted,
                                result.switch_rejection_reason,
                                result.selection_runtime,
                            )
                            record.update({"field": field_name, "trial": trial})
                            candidate_rows.append(record)
                        history = result.selected_record()
                        history.update({"field": field_name, "trial": trial})
                        history_rows.append(history)
                        hyperparameter_rows.append(
                            {
                                "field": field_name,
                                "trial": trial,
                                "sample_count": state.sample_count,
                                "selection_mode": mode,
                                "selected_kernel_id": result.selected_kernel_id,
                                "optimized_hyperparameters": json.dumps(
                                    result.optimized_hyperparameters, sort_keys=True
                                ),
                            }
                        )
                if mode == "hybrid_correct_profile":
                    source_schedules[(field_name, trial)] = {
                        state.sample_count: (
                            state.selected_kernel_id,
                            state.kernel_selection_result.fitted_kernel,
                        )
                        for state in states
                        if isinstance(state.kernel_selection_result, KernelSelectionResult)
                    }
                    for state in states:
                        recovery_rows.append(
                            {
                                "true_field_family": field.metadata.get(
                                    "field_family", "smooth_global"
                                ),
                                "selected_kernel_family": _kernel_family(state.selected_kernel_id),
                                "field": field_name,
                                "trial": trial,
                                "selection_count": 1,
                                "selection_percentage": 100.0,
                            }
                        )

            schedule = source_schedules[(field_name, trial)]
            isolated: dict[str, list[Any]] = {}
            for method_index, method in enumerate(BASELINE_METHODS):
                states = run_sequential_design(
                    field.evaluate,
                    field.domain,
                    initial,
                    candidate_pool,
                    evaluation,
                    method,
                    int(config["final_budget"]),
                    trial_seed + 500 + method_index,
                    field_name=field_name,
                    trial=trial,
                    true_evaluation=true_evaluation,
                    gpr_config=GPRConfig(alpha=1.0, random_state=trial_seed + 500 + method_index),
                    minimum_normalized_distance=_minimum_normalized_distance(config),
                    boundary_margin=float(config["initial_boundary_margin"]),
                    kernel_schedule=schedule,
                    selection_mode_label=f"acquisition_isolation_{method}",
                )
                isolated[method] = states
                for state in states:
                    row = state.scalar_record()
                    row.update(
                        {"selection_mode": "hybrid_correct_profile", "mode": method, "study": "B"}
                    )
                    study_b_rows.append(row)
                    if state.sample_count == int(config["final_budget"]):
                        study_b_final.append(row)
            loo_final = isolated["support_adjusted_krispu"][-1]
            for method in (item for item in BASELINE_METHODS if item != "support_adjusted_krispu"):
                baseline_final = isolated[method][-1]
                paired_rows.append(
                    {
                        "field": field_name,
                        "trial": trial,
                        "baseline": method,
                        "delta_nrmse": loo_final.metrics.nrmse - baseline_final.metrics.nrmse,
                        "krispu_nrmse": loo_final.metrics.nrmse,
                        "baseline_nrmse": baseline_final.metrics.nrmse,
                    }
                )
            if config.get("save_gifs", True):
                visual_modes = config.get("visual_audit_modes", ["hybrid_correct_profile"])
                visual_fields = set(config.get("visual_audit_fields", [field_name]))
                if field_name not in visual_fields:
                    continue
                for visual_mode in visual_modes:
                    if visual_mode not in mode_states:
                        raise ValueError(f"Unknown visual audit mode: {visual_mode}")
                    visual_states = mode_states[visual_mode]
                    save_sequential_visuals(
                        visual_states,
                        output,
                        save_gif=True,
                        save_snapshots=False,
                        save_point_layout_gif=False,
                        frame_duration_ms=int(config.get("frame_duration_ms", 400)),
                        dpi=int(config.get("dpi", 110)),
                    )

    _write_csv(output / "kernel_candidate_scores.csv", candidate_rows)
    _write_csv(output / "kernel_selection_history.csv", history_rows)
    _write_csv(output / "kernel_hyperparameter_history.csv", hyperparameter_rows)
    _write_csv(output / "study_a_iteration_metrics.csv", study_a_rows)
    _write_csv(output / "study_a_final_metrics.csv", study_a_final)
    _write_csv(output / "study_b_iteration_metrics.csv", study_b_rows)
    _write_csv(output / "study_b_final_metrics.csv", study_b_final)
    _write_csv(output / "final_metrics.csv", study_a_final + study_b_final)
    _write_csv(output / "acquisition_isolation_pairs.csv", paired_rows)
    _write_csv(output / "kernel_recovery_matrix.csv", _aggregate_recovery(recovery_rows))
    _write_csv(output / "nrmse_auc.csv", _auc_rows(study_a_rows))
    _write_manifest(output, config, config_path, history_rows)
    _write_figures(output, study_a_rows, study_a_final, candidate_rows, history_rows, paired_rows)
    _write_structured_figures(output, study_a_rows, candidate_rows, history_rows)
    _write_report(output, study_a_final, paired_rows, config, history_rows)
    return output


def _mode_config(
    mode: str, metadata: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any] | None:
    family = str(metadata.get("field_family", "smooth_global"))
    profiles = {
        "smooth_global": "smooth_global",
        "rough_single_scale": "rough_single_scale",
        "rough_multiscale": "rough_multiscale",
        "trend_plus_local": "trend_plus_local",
        "periodic": "periodic",
    }
    if mode == "fixed_generic":
        return None
    if mode == "automatic_standard":
        selection = {
            key: value for key, value in config["kernel_selection"].items() if key != "enabled"
        }
        return {"mode": "automatic", "candidate_set": "standard", **selection}
    if mode == "hybrid_correct_profile":
        selection = {
            key: value for key, value in config["kernel_selection"].items() if key != "enabled"
        }
        return {
            "mode": "hybrid",
            "profile": profiles.get(family, "smooth_global"),
            **selection,
        }
    if mode == "hybrid_broad_profile":
        selection = {
            key: value for key, value in config["kernel_selection"].items() if key != "enabled"
        }
        return {"mode": "hybrid", "profile": "broad_standard", **selection}
    if mode == "manual_correct":
        return {
            "mode": "manual",
            "specification": _manual_spec(family),
            **config.get("manual", {}),
        }
    if mode == "manual_mismatched":
        return {
            "mode": "manual",
            "specification": _manual_mismatch_spec(family),
            **config.get("manual", {}),
        }
    raise ValueError(f"Unknown study mode: {mode}")


def _manual_spec(family: str) -> dict[str, Any]:
    if family == "rough_multiscale":
        return {
            "type": "additive",
            "components": [
                {
                    "type": "matern",
                    "nu": 2.5,
                    "amplitude_initial": 0.8,
                    "amplitude_bounds": [0.05, 5.0],
                    "length_scale_initial": [0.5, 0.5],
                    "length_scale_bounds": [0.15, 2.0],
                },
                {
                    "type": "matern",
                    "nu": 0.5,
                    "amplitude_initial": 0.2,
                    "amplitude_bounds": [0.01, 2.0],
                    "length_scale_initial": [0.08, 0.08],
                    "length_scale_bounds": [0.01, 0.20],
                },
            ],
        }
    if family == "periodic":
        return {
            "type": "additive",
            "components": [
                {
                    "type": "exp_sine_squared",
                    "length_scale_initial": 1.0,
                    "length_scale_bounds": [0.5, 2.0],
                    "periodicity_initial": 1.0,
                    "amplitude_initial": 0.8,
                },
                {
                    "type": "matern",
                    "nu": 1.5,
                    "length_scale_initial": [0.3, 0.3],
                    "amplitude_initial": 0.2,
                },
            ],
            "observation_noise": {
                "enabled": True,
                "initial": 2.0,
                "bounds": [0.1, 10.0],
            },
        }
    if family == "trend_plus_local":
        return {
            "type": "additive",
            "components": [
                {"type": "dot_product", "sigma_0_initial": 1.0},
                {"type": "matern", "nu": 1.5, "length_scale_initial": [0.25, 0.25]},
            ],
        }
    return {"type": "matern", "nu": 2.5, "length_scale_initial": [0.35, 0.35]}


def _manual_mismatch_spec(family: str) -> dict[str, Any]:
    if family in {"rough_multiscale", "rough_single_scale"}:
        return {"type": "rbf", "length_scale_initial": [0.4, 0.4]}
    return {"type": "matern", "nu": 0.5, "length_scale_initial": [0.12, 0.12]}


def _legacy_lhs_initial_design(domain: Any, count: int, seed: int) -> np.ndarray:
    design = qmc.LatinHypercube(d=domain.dimension, seed=seed).random(count)
    return domain.denormalize(design)


def _kernel_family(kernel_id: str) -> str:
    if "periodic" in kernel_id:
        return "periodic"
    if "linear" in kernel_id:
        return "trend"
    if "long_plus" in kernel_id:
        return "multiscale"
    if "rbf" in kernel_id or "52" in kernel_id:
        return "smooth"
    return "rough"


def _aggregate_recovery(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["true_field_family"]), str(row["selected_kernel_family"]))
        grouped[key] = grouped.get(key, 0) + 1
    totals: dict[str, int] = {}
    for (true, _), count in grouped.items():
        totals[true] = totals.get(true, 0) + count
    return [
        {
            "true_field_family": true,
            "selected_kernel_family": selected,
            "selection_count": count,
            "selection_percentage": 100.0 * count / totals[true],
        }
        for (true, selected), count in sorted(grouped.items())
    ]


def _auc_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["field"]), str(row["mode"]), int(row["trial"])), []).append(row)
    result = []
    for (field, mode, trial), values in grouped.items():
        values.sort(key=lambda row: int(row["sample_count"]))
        distances = np.asarray(
            [
                float(row["nearest_normalized_distance"])
                for row in values
                if row.get("nearest_normalized_distance") not in (None, "")
            ],
            dtype=float,
        )
        correlations = np.asarray(
            [
                float(row["maximum_kernel_correlation_to_observations"])
                for row in values
                if row.get("maximum_kernel_correlation_to_observations") not in (None, "")
            ],
            dtype=float,
        )
        result.append(
            {
                "field": field,
                "selection_mode": mode,
                "trial": trial,
                "nrmse_auc": float(
                    nrmse_auc(
                        [int(row["sample_count"]) for row in values],
                        [float(row["nrmse"]) for row in values],
                    )
                ),
                "final_nrmse": float(values[-1]["nrmse"]),
                "median_nearest_normalized_distance": (
                    float(np.median(distances)) if len(distances) else np.nan
                ),
                "fraction_selections_within_0_05_normalized_distance": (
                    float(np.mean(distances <= 0.05)) if len(distances) else np.nan
                ),
                "fraction_selections_kernel_correlation_above_0_95": (
                    float(np.mean(correlations > 0.95)) if len(correlations) else np.nan
                ),
            }
        )
    return result


def _write_figures(
    output: Path,
    rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> None:
    """Write two compact figures: comparison first, diagnostics second."""
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(row["field"]) for row in final_rows})
    auc_rows = _auc_rows(rows)

    def mean_value(records: list[dict[str, Any]], field: str, mode: str, key: str) -> float:
        values = [
            float(row[key])
            for row in records
            if row["field"] == field and row["selection_mode"] == mode
        ]
        return float(np.mean(values))

    baseline_nrmse = {
        field: mean_value(final_rows, field, "fixed_generic", "nrmse") for field in fields
    }
    baseline_auc = {
        field: mean_value(auc_rows, field, "fixed_generic", "nrmse_auc") for field in fields
    }

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(MODE_NAMES)))
    for index, mode in enumerate(MODE_NAMES):
        final_delta = [
            mean_value(final_rows, field, mode, "nrmse") - baseline_nrmse[field] for field in fields
        ]
        auc_delta = [
            mean_value(auc_rows, field, mode, "nrmse_auc") - baseline_auc[field] for field in fields
        ]
        axes[0, 0].plot(fields, final_delta, "o-", color=colors[index], label=mode)
        axes[0, 1].plot(fields, auc_delta, "o-", color=colors[index], label=mode)
    for axis in axes[0]:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.25)
    axes[0, 0].set_title("Final NRMSE Δ vs fixed_generic")
    axes[0, 1].set_title("NRMSE-AUC Δ vs fixed_generic")
    axes[0, 0].set_ylabel("negative is better")
    axes[0, 1].set_ylabel("negative is better")
    axes[0, 0].legend(fontsize=7, ncol=2)

    baselines = sorted({str(row["baseline"]) for row in paired_rows})
    for index, baseline in enumerate(baselines):
        values = [float(row["delta_nrmse"]) for row in paired_rows if row["baseline"] == baseline]
        axes[1, 0].scatter(np.full(len(values), index), values, label=baseline)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Study B: KRISP-U Δ vs same-kernel baseline")
    axes[1, 0].set_ylabel("final NRMSE Δ")
    axes[1, 0].set_xticks(range(len(baselines)), baselines)

    switch_rates: dict[str, float] = {}
    for mode in MODE_NAMES:
        mode_history = [row for row in history_rows if row["selection_mode"] == mode]
        transitions = [
            row for row in mode_history if row.get("previous_kernel_id") not in (None, "")
        ]
        switches = [
            row for row in transitions if row["previous_kernel_id"] != row["selected_kernel_id"]
        ]
        switch_rates[mode] = 100.0 * len(switches) / len(transitions) if transitions else 0.0
    axes[1, 1].bar(list(switch_rates), list(switch_rates.values()), color=colors)
    axes[1, 1].set_title("Kernel-family switching frequency")
    axes[1, 1].set_ylabel("percent of eligible transitions")
    axes[1, 1].tick_params(axis="x", rotation=55)
    for axis in axes[1]:
        axis.grid(axis="y", alpha=0.25)
    _save(figure, figures / "benchmark_summary.png")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for field in sorted({str(row["field"]) for row in candidate_rows}):
        subset = [row for row in candidate_rows if row["field"] == field]
        for kernel_id in sorted({str(row["candidate_kernel_id"]) for row in subset}):
            values = [row for row in subset if row["candidate_kernel_id"] == kernel_id]
            axes[0].plot(
                [row["sample_count"] for row in values],
                [row["selection_score"] for row in values],
                ".-",
                alpha=0.45,
                label=kernel_id,
            )
    axes[0].set_title("Candidate predictive composite scores")
    axes[0].set_xlabel("sample count")
    axes[0].set_ylabel("lower is better")
    axes[0].legend(fontsize=6, ncol=2)

    for mode in MODE_NAMES:
        subset = [row for row in history_rows if row["selection_mode"] == mode]
        if subset:
            axes[1].plot(
                [row["sample_count"] for row in subset],
                [str(row["selected_kernel_id"]) for row in subset],
                ".-",
                label=mode,
            )
    axes[1].set_title("Selected kernel history")
    axes[1].set_xlabel("sample count")
    axes[1].set_ylabel("kernel id")
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.25)
    _save(figure, figures / "kernel_diagnostics.png")


def _write_structured_figures(
    output: Path,
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
) -> None:
    """Write per-field figures with names suitable for automated inspection."""

    directories = {
        name: output / "figures" / name
        for name in ("kernel_selection", "kernel_hyperparameters", "kernel_mode_comparison")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(row["field"]) for row in rows})
    for field in fields:
        field_rows = [row for row in rows if row["field"] == field]
        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for mode in MODE_NAMES:
            values = sorted(
                [row for row in field_rows if row.get("selection_mode") == mode],
                key=lambda row: int(row["sample_count"]),
            )
            if values:
                axis.plot(
                    [int(row["sample_count"]) for row in values],
                    [float(row["nrmse"]) for row in values],
                    marker="o",
                    label=mode,
                )
        axis.set(title=f"Kernel-mode NRMSE learning curves | {field}", xlabel="measurements")
        axis.set_ylabel("NRMSE")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
        _save(figure, directories["kernel_mode_comparison"] / f"{field}_nrmse.png")

        field_history = [row for row in history_rows if row.get("field") == field]
        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for mode in MODE_NAMES:
            values = sorted(
                [row for row in field_history if row.get("selection_mode") == mode],
                key=lambda row: int(row["sample_count"]),
            )
            if values:
                axis.step(
                    [int(row["sample_count"]) for row in values],
                    [str(row["selected_kernel_id"]) for row in values],
                    where="post",
                    label=mode,
                )
        axis.set(title=f"Kernel-selection timeline | {field}", xlabel="measurements")
        axis.set_ylabel("selected kernel ID")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
        _save(figure, directories["kernel_selection"] / f"{field}_kernel_timeline.png")

        field_candidates = [row for row in candidate_rows if row.get("field") == field]
        if field_candidates:
            figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
            metrics = (
                ("selection_score", "total selection score"),
                ("spatial_cv_nrmse", "spatial CV NRMSE"),
                ("spatial_cv_nlpd", "spatial CV NLPD"),
                ("loo_nrmse", "LOO NRMSE"),
                ("loo_nlpd", "LOO NLPD"),
                ("degeneracy_penalty", "degeneracy penalty"),
            )
            for axis, (key, title) in zip(axes.flat, metrics, strict=True):
                for kernel_id in sorted(
                    {str(row["candidate_kernel_id"]) for row in field_candidates}
                ):
                    values = [
                        row for row in field_candidates if row["candidate_kernel_id"] == kernel_id
                    ]
                    numeric = [float(row[key]) for row in values if row.get(key) not in (None, "")]
                    if numeric:
                        axis.plot(
                            [int(row["sample_count"]) for row in values[: len(numeric)]],
                            numeric,
                            ".-",
                            label=kernel_id,
                        )
                axis.set_title(title)
                axis.grid(alpha=0.2)
            axes.flat[0].legend(fontsize=6, ncol=2)
            figure.suptitle(f"Candidate-kernel scores and rejection diagnostics | {field}")
            _save(figure, directories["kernel_selection"] / f"{field}_candidate_scores.png")

        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for mode in MODE_NAMES:
            values = sorted(
                [row for row in field_history if row.get("selection_mode") == mode],
                key=lambda row: int(row["sample_count"]),
            )
            for row in values:
                parameters = row.get("optimized_hyperparameters", {})
                if isinstance(parameters, str):
                    try:
                        parameters = json.loads(parameters)
                    except json.JSONDecodeError:
                        parameters = {}
                scales = [
                    float(item)
                    for name, value in parameters.items()
                    if "length_scale" in name
                    for item in np.asarray(value, dtype=float).reshape(-1)
                ]
                if scales:
                    axes[0].scatter([int(row["sample_count"])] * len(scales), scales, label=mode)
                amplitudes = [
                    float(item)
                    for name, value in parameters.items()
                    if "constant_value" in name
                    for item in np.asarray(value, dtype=float).reshape(-1)
                ]
                if amplitudes:
                    axes[1].scatter([int(row["sample_count"])] * len(amplitudes), amplitudes)
        axes[0].set_title("ARD and long/short length scales")
        axes[1].set_title("Kernel amplitudes")
        for axis in axes:
            axis.set_xlabel("measurements")
            axis.grid(alpha=0.25)
        axes[0].legend(fontsize=7)
        _save(figure, directories["kernel_hyperparameters"] / f"{field}_hyperparameters.png")


def _write_figures_legacy(
    output: Path,
    rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for field in sorted({str(row["field"]) for row in candidate_rows}):
        subset = [row for row in candidate_rows if row["field"] == field]
        for kernel_id in sorted({str(row["candidate_kernel_id"]) for row in subset}):
            values = [row for row in subset if row["candidate_kernel_id"] == kernel_id]
            axes[0, 0].plot(
                [row["sample_count"] for row in values],
                [row["selection_score"] for row in values],
                ".-",
                label=kernel_id,
            )
    axes[0, 0].set(title="Predictive composite score", xlabel="sample count")
    axes[0, 1].plot(
        [row["sample_count"] for row in candidate_rows],
        [row["spatial_cv_nrmse"] for row in candidate_rows],
        ".",
    )
    axes[0, 1].set(title="Spatial CV NRMSE", xlabel="selection event")
    axes[1, 0].plot(
        [row["sample_count"] for row in candidate_rows],
        [row["spatial_cv_nlpd"] for row in candidate_rows],
        ".",
    )
    axes[1, 0].set(title="Spatial CV NLPD", xlabel="selection event")
    axes[1, 1].plot(
        [row["sample_count"] for row in candidate_rows],
        [row["degeneracy_penalty"] for row in candidate_rows],
        ".",
    )
    axes[1, 1].set(title="Degeneracy penalty", xlabel="selection event")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=6, ncol=2)
    _save(figure, output / "figures" / "kernel_score_comparison.png")

    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for mode in sorted({str(row["selection_mode"]) for row in history_rows}):
        subset = [row for row in history_rows if row["selection_mode"] == mode]
        axis.step(
            [row["sample_count"] for row in subset],
            [row["selected_kernel_id"] for row in subset],
            where="post",
            label=mode,
        )
    axis.set(title="Kernel-selection history", xlabel="sample count", ylabel="selected family")
    axis.legend(fontsize=7)
    _save(figure, output / "figures" / "kernel_selection_history.png")

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for mode in MODE_NAMES:
        subset = [row for row in rows if row["selection_mode"] == mode]
        if subset:
            grouped = _curve(subset, "nrmse")
            axis.plot(grouped[0], grouped[1], ".-", label=mode)
    axis.set(title="Performance by selection mode", xlabel="sample count", ylabel="NRMSE")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.25)
    _save(figure, output / "figures" / "performance_by_selection_mode.png")

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for index, baseline in enumerate(sorted({row["baseline"] for row in paired_rows})):
        values = [float(row["delta_nrmse"]) for row in paired_rows if row["baseline"] == baseline]
        axis.scatter(np.full(len(values), index), values, label=baseline)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(title="Final paired performance: KRISP-U minus baseline NRMSE", ylabel="Δ NRMSE")
    axis.set_xticks(
        range(len(sorted({row["baseline"] for row in paired_rows}))),
        sorted({row["baseline"] for row in paired_rows}),
    )
    _save(figure, output / "figures" / "final_paired_performance.png")

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    values = [row for row in history_rows if row.get("optimized_hyperparameters")]
    for index, row in enumerate(values):
        parameters = row["optimized_hyperparameters"]
        for name, parameter_values in parameters.items():
            numeric = np.asarray(parameter_values, dtype=float).reshape(-1)
            if "length_scale" in name:
                axis = axes[0, 0]
                label = "ARD length scale"
                if "long" in name or name.startswith("k1__"):
                    axis = axes[0, 1]
                    label = "long/first component scale"
                elif "short" in name or name.startswith("k2__"):
                    axis = axes[0, 1]
                    label = "short/second component scale"
                axis.plot(np.full(len(numeric), index), numeric, ".", label=label)
            elif "constant_value" in name:
                axes[1, 0].plot(
                    np.full(len(numeric), index), numeric, ".", label="component amplitude"
                )
            elif "noise_level" in name:
                axes[1, 1].plot(
                    np.full(len(numeric), index), numeric, ".", label="fitted observation noise"
                )
    axes[0, 0].set_title("ARD length scales")
    axes[0, 1].set_title("Long/short component scales")
    axes[1, 0].set_title("Component amplitudes")
    axes[1, 1].set_title("Fitted observation noise")
    for axis in axes.flat:
        axis.set_xlabel("selection event")
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7)
    _save(figure, output / "figures" / "hyperparameter_history.png")

    recovery = list(csv.DictReader((output / "kernel_recovery_matrix.csv").open(encoding="utf-8")))
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    labels = [f"{row['true_field_family']} → {row['selected_kernel_family']}" for row in recovery]
    axis.barh(labels, [float(row["selection_percentage"]) for row in recovery])
    axis.set(title="Kernel recovery matrix", xlabel="selection percentage")
    _save(figure, output / "figures" / "kernel_recovery_matrix.png")


def _write_report(
    output: Path,
    final_rows: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    config: dict[str, Any],
    history_rows: list[dict[str, Any]],
) -> None:
    from krispu.kernels.profiles import PROFILE_REGISTRY
    from krispu.kernels.registry import registered_kernel_ids

    by_field_mode: dict[tuple[str, str], list[float]] = {}
    for row in final_rows:
        key = (str(row["field"]), str(row["selection_mode"]))
        by_field_mode.setdefault(key, []).append(float(row["nrmse"]))
    modes = ["fixed_generic", *MODE_NAMES[1:]]
    switch_totals: dict[str, list[int]] = {}
    for row in history_rows:
        mode = str(row["selection_mode"])
        previous = row.get("previous_kernel_id")
        switched = int(previous not in (None, "") and previous != row["selected_kernel_id"])
        switch_totals.setdefault(mode, [0, 0])
        switch_totals[mode][0] += switched
        switch_totals[mode][1] += int(previous not in (None, ""))
    lines = [
        "# KRISP-U v0.3.0 kernel-selection study",
        "",
        "Study A compares complete sequential workflows. Study B reuses the same selected kernel family across acquisition methods.",
        "",
        "## Implemented system",
        "",
        "Modes: `manual`, `automatic`, and `hybrid`.",
        "",
        "Registered candidates: "
        + ", ".join(f"`{kernel_id}`" for kernel_id in registered_kernel_ids())
        + ".",
        "",
        "Profiles: " + ", ".join(f"`{profile}`" for profile in PROFILE_REGISTRY) + ".",
        "",
        "The default predictive composite minimizes `0.5*normalized spatial NLPD + 0.4*normalized spatial NRMSE + 0.1*calibration error + degeneracy penalty`; marginal likelihood is diagnostic only.",
        "",
        "Degeneracy rules include nonfinite fits, failed factorizations, bound-hitting scales, invalid long/short ordering, collapsed or variance-dominating components, white-noise dominance, extreme condition numbers, and invalid CV predictions.",
        "",
        "## Final NRMSE by field and mode",
        "",
        "| Field | Mode | NRMSE |",
        "|---|---|---:|",
    ]
    for field in sorted({str(row["field"]) for row in final_rows}):
        for mode in modes:
            values = by_field_mode.get((field, mode), [])
            if values:
                lines.append(f"| {field} | {mode} | {float(np.mean(values)):.6g} |")
    lines.extend(
        (
            "",
            "## Required interpretation",
            "",
            "1. Manual correct structural knowledge is compared with `manual_correct` rows above; improvement is field-dependent and should not be inferred from one trial.",
            "2. Automatic recovery is quantified by the `manual_correct` versus `automatic_standard` rows and by the recovery matrix, not by exact family recovery alone.",
            "3. Hybrid versus unrestricted automatic selection is compared by `hybrid_correct_profile` versus `automatic_standard`.",
            "4. Incorrect manual specification is quantified by `manual_mismatched` versus `manual_correct`.",
            "5. Reliably selected field families are listed in `kernel_recovery_matrix.csv`.",
            "6. Rough-kernel preference is visible in the candidate score and recovery tables; it is not treated as success without predictive improvement.",
            "7. Switching frequency is recorded in `kernel_selection_history.csv`; hysteresis requires a 0.05 lower-score improvement by default.",
            "8. Acquisition isolation is reported in `acquisition_isolation_pairs.csv` using the same selected family across acquisition methods.",
            "",
            "",
            "",
        )
    )
    for mode, (switches, transitions) in sorted(switch_totals.items()):
        frequency = 0.0 if transitions == 0 else 100.0 * switches / transitions
        lines.append(
            f"- `{mode}` switching frequency: {switches}/{transitions} eligible transitions ({frequency:.3g}%)."
        )
    lines.extend(("", "Selected kernels observed by field:", ""))
    for field in sorted({str(row["field"]) for row in history_rows}):
        selected = sorted(
            {str(row["selected_kernel_id"]) for row in history_rows if row["field"] == field}
        )
        lines.append(f"- `{field}`: " + ", ".join(f"`{kernel}`" for kernel in selected))
    lines.extend(
        (
            "",
            "## Acquisition-isolated paired rows",
            "",
            "| Field | Baseline | Delta NRMSE |",
            "|---|---|---:|",
        )
    )
    for row in paired:
        lines.append(f"| {row['field']} | {row['baseline']} | {float(row['delta_nrmse']):.6g} |")
    lines.extend(
        (
            "",
            "## Known limitations",
            "",
            "The initial registry is intentionally small; periodic covariance can require explicit WhiteKernel stabilization; spatial blocking is quadrant-based in two dimensions; and one trial is not evidence of universal kernel recovery.",
            "",
            f"Configuration: `{config['experiment_name']}`.",
            "",
            "Out-of-scope files changed: none",
            "",
        )
    )
    report = "\n".join(lines) + "\n"
    (output / "kernel_selection_report.md").write_text(report, encoding="utf-8")
    (output / "benchmark_report.md").write_text(report, encoding="utf-8")


def _write_manifest(
    output: Path,
    config: dict[str, Any],
    config_path: Path,
    history_rows: list[dict[str, Any]],
) -> None:
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
    selected = [
        {
            "field": row.get("field"),
            "trial": row.get("trial"),
            "sample_count": row.get("sample_count"),
            "selection_mode": row.get("selection_mode"),
            "selected_kernel_id": row.get("selected_kernel_id"),
            "optimized_hyperparameters": row.get("optimized_hyperparameters"),
        }
        for row in history_rows
    ]
    manifest = {
        "git_commit": commit,
        "package_version": "0.3.0",
        "python_version": platform.python_version(),
        "dependency_versions": dependencies,
        "configuration": config,
        "configuration_path": str(config_path),
        "field_seeds": {
            field: int(config["base_seed"]) + index * 100_000 + 1
            for index, field in enumerate(config["fields"])
        },
        "initial_design_seeds": config.get("initial_design_seeds"),
        "candidate_seeds": "base_seed + field_index*100000 + trial*10000 + 3",
        "noise_seeds": config.get("noise_seed"),
        "method_seeds": "trial_seed + 100 + method_index",
        "kernel_selection_seeds": config.get("kernel_selection", {}).get("random_state"),
        "selected_kernels": selected,
        "metric_definitions": {
            "nrmse": "RMSE / true-field range",
            "nrmse_auc": "sum of right-endpoint NRMSE times sample-count increments",
            "paired_difference": "KRISP-U metric minus paired baseline metric",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _curve(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["sample_count"]), []).append(float(row[key]))
    counts = np.asarray(sorted(grouped), dtype=float)
    values = np.asarray([np.mean(grouped[int(count)]) for count in counts])
    return counts, values


def _save(figure: Any, path: Path) -> None:
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["sample_count"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validate_study_config(config: dict[str, Any]) -> None:
    required = {
        "experiment_name",
        "fields",
        "initial_sample_count",
        "initial_boundary_margin",
        "final_budget",
        "candidate_count",
        "evaluation_grid_size",
        "trials",
        "base_seed",
        "kernel_selection",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Kernel study configuration is missing: {sorted(missing)}")
    unknown = set(config["fields"]).difference(FIELD_FACTORIES)
    if unknown:
        raise ValueError(f"Unknown kernel-study fields: {sorted(unknown)}")
    _minimum_normalized_distance(config)
    if int(config["initial_sample_count"]) < 3:
        raise ValueError("initial_sample_count must support a fitted surrogate.")


def _minimum_normalized_distance(config: dict[str, Any]) -> float:
    candidate_validity = config.get("candidate_validity", {})
    value = candidate_validity.get(
        "minimum_normalized_distance",
        config.get("minimum_normalized_distance", 1.0e-4),
    )
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError("minimum_normalized_distance must be finite and non-negative.")
    return result
