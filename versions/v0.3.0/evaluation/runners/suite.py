"""CLI for reproducible v0.3.0 visual and paired performance audits."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evaluation.fields import FIELD_FACTORIES
from evaluation.figures.field_plots import (
    plot_error_concentration,
    plot_field_audit,
    plot_uncertainty_components,
    plot_uncertainty_error,
)
from evaluation.figures.sequential import save_sequential_visuals
from evaluation.runners.config import (
    jackknife_config,
    load_config,
    minimum_normalized_distance,
    prepare_suite_output,
    write_resolved_config,
)
from evaluation.runners.design import initial_design, make_field, regular_grid
from evaluation.runners.output import RECORD_FIELDS, save_spatial_state, write_records
from evaluation.runners.sequential import run_sequential_design
from krispu.candidates import generate_candidates
from krispu.config import GPRConfig
from krispu.domains import CandidateDomain, ContinuousDomain
from krispu.kernels.builders import build_kernel_from_spec, build_named_kernel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    output = run_benchmark(args.config)
    print(output)


def run_benchmark(config_path: Path, output_root: Path = Path("outputs")) -> Path:
    config = load_config(config_path)
    if config.get("study") == "kernel_selection":
        from evaluation.runners.kernel_study import run_kernel_selection_study

        return run_kernel_selection_study(config_path, output_root, config=config)
    if config.get("study") == "acquisition_comparison":
        from evaluation.runners.acquisition_study import run_acquisition_comparison_study

        return run_acquisition_comparison_study(config, output_root)
    if not config.get("include_noisy_field", True):
        config["fields"] = [
            field for field in config["fields"] if field not in {"noisy", "noisy_baseline"}
        ]
    config["methods"] = [
        "support_adjusted_krispu" if method == "krispu" else method for method in config["methods"]
    ]
    _validate_config(config)
    min_distance = minimum_normalized_distance(config)
    output = prepare_suite_output(output_root, config["experiment_name"])
    write_resolved_config(output, config)
    mode = config["output"]["mode"]
    for directory in (
        output / "metrics",
        output / "kernel",
        output / "figures" / "fields",
        output / "figures" / "global",
        output / "animations",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    figure_dirs = {
        name: output / "figures" / name
        for name in (
            "field_audits",
            "uncertainty_components",
            "learning_curves",
            "sampling_paths",
            "uncertainty_error",
            "component_evolution",
            "paired_performance",
            "boundary_diagnostics",
            "dominant_jackknife",
        )
    }
    if mode != "summary":
        for directory in figure_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        (output / "spatial_arrays").mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    final_records: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    seeds: list[dict[str, int | str]] = []
    final_by_field: dict[str, dict[str, Any]] = {}
    field_states: dict[str, dict[str, list[Any]]] = {}
    component_records: list[dict[str, Any]] = []
    diagnostic_states: dict[str, list[Any]] = {}
    kernel_events: list[dict[str, Any]] = []
    candidate_scores: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    for field_index, field_name in enumerate(config["fields"]):
        field_seed = int(config["base_seed"]) + field_index * 100_000 + 1
        field = make_field(field_name, field_seed)
        evaluation = regular_grid(field.domain, int(config["evaluation_grid_size"]))
        field_final: dict[str, Any] = {}
        for trial in range(int(config["trials"])):
            trial_seed = int(config["base_seed"]) + field_index * 100_000 + trial * 10_000
            field_seed = trial_seed + 1
            initial_seed_list = config.get("initial_design_seeds")
            initial_design_seed = (
                int(initial_seed_list[trial % len(initial_seed_list)])
                if initial_seed_list
                else trial_seed + 2
            )
            candidate_seed = trial_seed + 3
            candidate_pool = generate_candidates(
                field.domain, int(config["candidate_count"]), "lhs", candidate_seed
            )
            initial_X, initial_jackknife_eligible = initial_design(
                config["initial_design"],
                field.domain,
                int(config["initial_sample_count"]),
                float(config["initial_boundary_margin"]),
                initial_design_seed,
                return_eligibility=True,
            )
            if field.metadata.get("observation_design") == "clustered":
                initial_X, initial_jackknife_eligible = initial_design(
                    "clustered_observations",
                    field.domain,
                    int(config["initial_sample_count"]),
                    float(config["initial_boundary_margin"]),
                    initial_design_seed,
                    return_eligibility=True,
                )
            true_evaluation = field.evaluate(evaluation)
            seeds.append(
                {
                    "field": field_name,
                    "trial": trial,
                    "field_seed": field_seed,
                    "initial_design_seed": initial_design_seed,
                    "candidate_seed": candidate_seed,
                }
            )
            for method_index, requested_method in enumerate(config["methods"]):
                method_seed = trial_seed + 100 + method_index
                method, method_config, selection_config = _algorithm_settings(
                    requested_method, config, method_seed, field.domain.dimension
                )
                seeds[-1][f"{requested_method}_method_seed"] = method_seed
                states = run_sequential_design(
                    field.evaluate,
                    field.domain,
                    initial_X,
                    candidate_pool,
                    evaluation,
                    method,
                    int(config["final_budget"]),
                    method_seed,
                    field_name=field_name,
                    trial=trial,
                    true_evaluation=true_evaluation,
                    gpr_config=method_config,
                    initial_jackknife_eligible=initial_jackknife_eligible,
                    minimum_normalized_distance=min_distance,
                    boundary_margin=float(config["initial_boundary_margin"]),
                    kernel_selection_config=selection_config,
                    selection_mode_label=requested_method,
                )
                final_state = states[-1]
                field_final[requested_method] = final_state
                field_states.setdefault(field_name, {})[requested_method] = states
                for state in states:
                    record = state.scalar_record
                    record["method"] = requested_method
                    record["study"] = config.get("study", "reconstruction_performance")
                    all_records.append(record)
                    if state.sample_count == int(config["final_budget"]):
                        final_records.append(record)
                    component_records.append(record)
                    if mode in {"diagnostic", "debug"}:
                        array_name = f"{field_name}_trial{trial}_{method}_n{state.sample_count}.npz"
                        save_spatial_state(output / "spatial_arrays" / array_name, state)
                    selection_result = getattr(state, "kernel_selection_result", None)
                    if selection_result is not None:
                        kernel_events.append(
                            {
                                "field": field_name,
                                "trial": trial,
                                "sample_count": state.sample_count,
                                "optimization_completed": selection_result.optimization_event.hyperparameters_optimized,
                                "reselection_triggered": selection_result.reselection_event.reselection_triggered,
                                "reselection_reasons": ";".join(
                                    selection_result.reselection_event.reselection_reasons
                                ),
                                "previous_family": selection_result.previous_kernel_id,
                                "selected_family": selection_result.selected_kernel_id,
                                "switch_accepted": selection_result.switch_accepted,
                                "length_scales": ";".join(
                                    str(value)
                                    for value in selection_result.optimization_event.current_length_scales
                                ),
                                "validation_score": selection_result.selection_score,
                                "best_challenger": getattr(
                                    selection_result.reselection_event,
                                    "best_challenger_kernel_id",
                                    None,
                                ),
                                "challenger_score": selection_result.reselection_event.challenger_validation_score,
                                "score_improvement": selection_result.reselection_event.score_improvement,
                                "optimization_runtime": selection_result.optimization_event.optimization_runtime,
                                "reselection_runtime": selection_result.reselection_event.reselection_runtime,
                            }
                        )
                        if selection_result.reselection_event.reselection_triggered:
                            candidate_scores.extend(
                                score.as_record(
                                    state.sample_count,
                                    selection_result.previous_kernel_id,
                                    selection_result.selected_kernel_id,
                                    selection_result.switch_accepted,
                                )
                                | {"field": field_name, "trial": trial}
                                for score in selection_result.candidate_scores
                            )
                if requested_method == "krispu_adaptive" and field.metadata.get("true_kernel"):
                    truth = field.metadata["true_kernel"]
                    adaptive_events = [
                        row
                        for row in kernel_events
                        if row["field"] == field_name and row["trial"] == trial
                    ]
                    final_scales = np.asarray(states[-1].current_length_scales, dtype=float)
                    true_scales = np.asarray(truth["ard_length_scales"], dtype=float)
                    lower_contacts = [
                        _state_touches_scale_bound(state)
                        for state in states
                    ]
                    recovery_rows.append(
                        {
                            "field": field_name,
                            "trial": trial,
                            "true_family": truth["family"],
                            "recovered_family": states[-1].selected_kernel_id,
                            "family_recovered": states[-1].selected_kernel_id == truth["family"],
                            "length_scale_relative_error": float(
                                np.mean(np.abs(final_scales - true_scales) / true_scales)
                            ),
                            "scale_bound_contact_rate": float(np.mean(lower_contacts)),
                            "reselection_count": sum(
                                str(row["reselection_triggered"]).lower() == "true"
                                for row in adaptive_events
                            ),
                            "accepted_switch_count": sum(
                                str(row["switch_accepted"]).lower() == "true"
                                and row.get("previous_family")
                                not in (None, "", row.get("selected_family"))
                                for row in adaptive_events
                            ),
                            "sample_count_at_final_switch": max(
                                (
                                    int(row["sample_count"])
                                    for row in adaptive_events
                                    if str(row["switch_accepted"]).lower() == "true"
                                    and row.get("previous_family")
                                    not in (None, "", row.get("selected_family"))
                                ),
                                default=None,
                            ),
                            "buffered_validation_score": states[-1].selection_score,
                        }
                    )
                if mode != "summary" and (
                    config.get("save_gifs", False) or config.get("save_png_snapshots", False)
                ):
                    save_sequential_visuals(
                        states,
                        output,
                        save_gif=bool(config.get("save_gifs", False)),
                        save_snapshots=(
                            mode == "debug" and bool(config.get("save_png_snapshots", False))
                        ),
                        snapshot_every=config.get("snapshot_every"),
                        snapshot_sample_counts=config.get("snapshot_sample_counts"),
                        frame_duration_ms=int(config.get("frame_duration_ms", 500)),
                        final_frame_duration_ms=int(config.get("final_frame_duration_ms", 1500)),
                        dpi=int(config.get("dpi", 150)),
                        annotate_point_order=bool(config.get("annotate_point_order", True)),
                        save_point_layout_gif=bool(
                            config.get("save_point_layout_animations", False)
                        ),
                        save_snapshot_gif=bool(config.get("save_snapshot_gifs", True)),
                        save_contact_sheet=bool(config.get("save_contact_sheet", True)),
                    )
                if mode != "summary" and method == "support_adjusted_krispu" and trial == 0:
                    diagnostic_states.setdefault(field_name, []).extend(states)
                    snapshot_counts = {
                        int(value) for value in (config.get("snapshot_sample_counts") or [])
                    }
                    for state in states:
                        if state.sample_count in snapshot_counts:
                            plot_field_audit(
                                state,
                                figure_dirs["field_audits"]
                                / f"{field_name}_n{state.sample_count}.png",
                                bool(config.get("save_pdf", False)),
                            )
                if mode != "summary" and method == "support_adjusted_krispu" and trial == 0:
                    mid_state = states[min(len(states) - 1, max(0, len(states) // 2))]
                    plot_uncertainty_components(
                        mid_state,
                        figure_dirs["uncertainty_components"] / f"{field_name}_components.png",
                        bool(config.get("save_pdf", False)),
                    )
                    plot_uncertainty_error(
                        mid_state,
                        figure_dirs["uncertainty_error"] / f"{field_name}_uncertainty_error.png",
                        bool(config.get("save_pdf", False)),
                    )
                    plot_error_concentration(
                        mid_state,
                        figure_dirs["uncertainty_error"] / f"{field_name}_error_concentration.png",
                        bool(config.get("save_pdf", False)),
                    )
            for baseline in (
                "raw_jackknife_sensitivity",
                "posterior_std",
                "random",
                "lhs",
                "maximin",
            ):
                if baseline in field_final and "support_adjusted_krispu" in field_final:
                    paired_rows.append(
                        {
                            "field": field_name,
                            "trial": trial,
                            "baseline": baseline,
                            "delta_nrmse": field_final["support_adjusted_krispu"].metrics.nrmse
                            - field_final[baseline].metrics.nrmse,
                            "krispu_nrmse": field_final["support_adjusted_krispu"].metrics.nrmse,
                            "baseline_nrmse": field_final[baseline].metrics.nrmse,
                        }
                    )
        final_by_field[field_name] = field_final
    summary_records = _summary_records(all_records)
    if mode == "summary":
        from evaluation.figures.summary import write_summary_figures

        write_summary_figures(
            field_states,
            all_records,
            output,
            dpi=int(config.get("dpi", 150)),
        )
    write_records(output / "metrics" / "per_step.csv", all_records, RECORD_FIELDS)
    write_records(output / "metrics" / "final.csv", final_records, RECORD_FIELDS)
    write_records(
        output / "metrics" / "aggregate.csv",
        summary_records,
        [
            "field",
            "method",
            "trial",
            "final_nrmse",
            "nrmse_auc",
            "final_r2",
            "p95_absolute_error",
            "uncertainty_error_rank_correlation",
            "high_error_region_capture",
            "near_neighbor_acquisition_rate",
            "runtime",
            "reselection_count",
            "accepted_switch_count",
            "fraction_near_boundary",
            "fraction_hull_vertices",
            "median_nearest_observation_distance",
            "fraction_selections_within_0_05_normalized_distance",
            "fraction_selections_kernel_correlation_above_0_95",
        ],
    )
    manifest = _manifest(config, config_path, seeds)
    (output / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    write_records(
        output / "kernel" / "events.csv",
        kernel_events,
        [
            "field",
            "trial",
            "sample_count",
            "optimization_completed",
            "reselection_triggered",
            "reselection_reasons",
            "previous_family",
            "selected_family",
            "switch_accepted",
            "length_scales",
            "validation_score",
            "best_challenger",
            "challenger_score",
            "score_improvement",
            "optimization_runtime",
            "reselection_runtime",
        ],
    )
    score_fields = sorted({key for row in candidate_scores for key in row}) or [
        "field",
        "trial",
        "sample_count",
        "candidate_kernel_id",
        "selection_score",
    ]
    write_records(output / "kernel" / "candidate_scores.csv", candidate_scores, score_fields)
    if recovery_rows:
        family_totals: dict[str, int] = {}
        family_hits: dict[str, int] = {}
        for row in recovery_rows:
            family = str(row["true_family"])
            family_totals[family] = family_totals.get(family, 0) + 1
            family_hits[family] = family_hits.get(family, 0) + int(
                bool(row["family_recovered"])
            )
        for row in recovery_rows:
            family = str(row["true_family"])
            row["family_recovery_frequency"] = family_hits[family] / family_totals[family]
        write_records(
            output / "kernel" / "recovery_summary.csv", recovery_rows, list(recovery_rows[0])
        )
    from evaluation.reports.benchmark_report import generate_report

    generate_report(output)
    return output


def _prepare_benchmark_output(output_root: Path, experiment_name: str) -> Path:
    """Replace one named run while preserving sibling studies in the root."""
    output_root = output_root.resolve()
    if output_root == Path.cwd().resolve() or output_root.parent == output_root:
        raise ValueError("output_root must be a dedicated directory")
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / experiment_name
    if output.exists():
        if output.is_dir() and not output.is_symlink():
            shutil.rmtree(output, onerror=_remove_readonly)
        else:
            output.unlink()
    output.mkdir(parents=True, exist_ok=True)
    return output


def _remove_readonly(function: Any, path: str, _exc_info: Any) -> None:
    """Retry generated-file cleanup after clearing a Windows read-only bit."""
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "experiment_name",
        "fields",
        "methods",
        "initial_design",
        "initial_sample_count",
        "initial_boundary_margin",
        "final_budget",
        "candidate_count",
        "evaluation_grid_size",
        "trials",
        "base_seed",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Benchmark configuration is missing: {sorted(missing)}")
    unknown_fields = set(config["fields"]).difference(FIELD_FACTORIES)
    if unknown_fields:
        raise ValueError(f"Unknown fields: {sorted(unknown_fields)}")
    study = str(config.get("study", "reconstruction_performance"))
    if study not in {"reconstruction_performance", "kernel_recovery", "kernel_selection", "acquisition_comparison"}:
        raise ValueError(f"Unknown study: {study}")
    unknown_methods = set(config["methods"]).difference(
        {
            "support_adjusted_krispu",
            "raw_jackknife_sensitivity",
            "posterior_std",
            "random",
            "lhs",
            "maximin",
            "krispu_fixed_gaussian",
            "krispu_fixed_matern32",
            "krispu_manual",
            "krispu_adaptive",
            "gp_posterior_variance",
            "random_sequential",
        }
    )
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")
    if int(config["initial_sample_count"]) < 3:
        raise ValueError("initial_sample_count must support a fitted surrogate.")
    if config["initial_design"] not in {
        "interior_maximin",
        "random_interior",
        "lhs_interior",
        "anchored_boundary",
        "clustered_observations",
    }:
        raise ValueError("Unknown initial_design.")
    if not 0 <= float(config["initial_boundary_margin"]) < 0.5:
        raise ValueError("initial_boundary_margin must be in [0, 0.5).")
    if _minimum_normalized_distance(config) < 0:
        raise ValueError("minimum_normalized_distance must be non-negative.")
    if int(config["final_budget"]) < int(config["initial_sample_count"]):
        raise ValueError("final_budget must be at least initial_sample_count.")
    if config.get("snapshot_every") is not None and int(config["snapshot_every"]) <= 0:
        raise ValueError("snapshot_every must be positive when provided.")
    if int(config.get("frame_duration_ms", 500)) <= 0:
        raise ValueError("frame_duration_ms must be positive.")
    if int(config.get("dpi", 150)) <= 0:
        raise ValueError("dpi must be positive.")


def _algorithm_settings(
    method: str,
    config: dict[str, Any],
    random_state: int,
    dimension: int,
) -> tuple[str, GPRConfig, dict[str, Any] | None]:
    """Resolve one named comparator into the existing sequential core."""

    base = GPRConfig(random_state=random_state, jackknife=jackknife_config(config))
    if method == "krispu_fixed_gaussian":
        return (
            "support_adjusted_krispu",
            replace(
                base,
                kernel=build_named_kernel("gaussian_ard", dimension, False),
                optimize_hyperparameters=False,
            ),
            None,
        )
    if method == "krispu_fixed_matern32":
        return (
            "support_adjusted_krispu",
            replace(
                base,
                kernel=build_named_kernel("matern_32_ard", dimension, False),
                optimize_hyperparameters=False,
            ),
            None,
        )
    if method == "krispu_manual":
        manual_scales = ([0.15, 0.45] + [0.25] * max(0, dimension - 2))[:dimension]
        specification = {
            "type": "matern_32_ard",
            "length_scale_initial": manual_scales,
            "length_scale_bounds": [0.02, 2.0],
        }
        return (
            "support_adjusted_krispu",
            replace(
                base,
                kernel=build_kernel_from_spec(specification, dimension, True),
                optimize_hyperparameters=True,
            ),
            None,
        )
    if method == "krispu_adaptive":
        selection = dict(config.get("kernel_selection", {}))
        selection.setdefault("mode", "automatic")
        return "support_adjusted_krispu", base, selection
    if method == "gp_posterior_variance":
        return "posterior_std", base, None
    if method == "random_sequential":
        return "random", base, None
    return (
        method,
        replace(
            base,
            optimize_hyperparameters=bool(config.get("optimize_hyperparameters", True)),
        ),
        None,
    )


def _initial_design(
    name: str,
    domain: CandidateDomain | None = None,
    sample_count: int = 5,
    boundary_margin: float = 0.05,
    random_state: int = 0,
    return_eligibility: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return a deterministic initial design and its explicit jackknife mask."""

    if domain is None:
        domain = ContinuousDomain([[-1.0, 1.0], [-1.0, 1.0]])
    if domain.dimension != 2 or sample_count != 5:
        raise ValueError("The v0.3.0 square audit requires five two-dimensional points.")
    if name == "anchored_boundary":
        normalized = np.asarray(
            [[0.0, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
            dtype=float,
        )
        design = domain.denormalize(normalized)
        eligibility = np.array([False, True, True, True, True])
        return (design, eligibility) if return_eligibility else design
    if name in {"random_interior", "lhs_interior"}:
        candidates = generate_candidates(domain, max(100, sample_count * 50), "lhs", random_state)
        normalized = domain.normalize(candidates)
        interior = candidates[
            np.all((normalized >= boundary_margin) & (normalized <= 1.0 - boundary_margin), axis=1)
        ]
        if len(interior) < sample_count:
            raise ValueError("The interior candidate set is too small for the initial design.")
        if name == "random_interior":
            selected = np.random.default_rng(random_state).choice(
                len(interior), size=sample_count, replace=False
            )
        else:
            selected = np.arange(sample_count)
        design = interior[np.asarray(selected)].copy()
        eligibility = np.ones(sample_count, dtype=bool)
        return (design, eligibility) if return_eligibility else design
    if name != "interior_maximin":
        raise ValueError("Unknown initial design.")
    oversampled = generate_candidates(domain, max(500, sample_count * 100), "lhs", random_state)
    normalized = domain.normalize(oversampled)
    interior = oversampled[
        np.all((normalized >= boundary_margin) & (normalized <= 1.0 - boundary_margin), axis=1)
    ]
    if len(interior) < sample_count:
        raise ValueError("The interior candidate set is too small for the initial design.")
    normalized = domain.normalize(interior)
    selected = [int(np.argmin(np.linalg.norm(normalized - 0.5, axis=1)))]
    while len(selected) < sample_count:
        distances = np.linalg.norm(
            normalized[:, None, :] - normalized[np.asarray(selected)][None, :, :], axis=2
        ).min(axis=1)
        distances[selected] = -np.inf
        selected.append(int(np.argmax(distances)))
    design = interior[np.asarray(selected)].copy()
    eligibility = np.ones(sample_count, dtype=bool)
    return (design, eligibility) if return_eligibility else design


def _make_field(field_name: str, seed: int) -> Any:
    """Construct a field with its stored seed when the factory supports one."""

    factory = FIELD_FACTORIES[field_name]
    try:
        return factory(seed=seed)
    except TypeError:
        return factory()


def _summary_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault((str(row["field"]), str(row["method"]), int(row["trial"])), []).append(
            row
        )
    summaries: list[dict[str, Any]] = []
    for (field, method, trial), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["sample_count"]))
        counts = np.asarray([int(row["sample_count"]) for row in rows], dtype=float)
        nrmse = np.asarray([float(row["nrmse"]) for row in rows], dtype=float)
        boundary = [
            row["near_domain_boundary"]
            for row in rows
            if row["near_domain_boundary"] not in (None, "")
        ]
        hull = [
            row["on_current_sample_hull"]
            for row in rows
            if row["on_current_sample_hull"] not in (None, "")
        ]
        distances = [
            float(row["nearest_normalized_distance"])
            for row in rows
            if row["nearest_normalized_distance"] not in (None, "")
        ]
        correlations = [
            float(row["maximum_kernel_correlation_to_observations"])
            for row in rows
            if row["maximum_kernel_correlation_to_observations"] not in (None, "")
        ]
        summaries.append(
            {
                "field": field,
                "method": method,
                "trial": trial,
                "final_nrmse": float(rows[-1]["nrmse"]),
                "nrmse_auc": float(np.sum(nrmse[1:] * np.diff(counts))),
                "final_r2": float(rows[-1]["r2"]),
                "p95_absolute_error": float(rows[-1]["p95_absolute_error"]),
                "uncertainty_error_rank_correlation": _mean_numeric(
                    rows, "uncertainty_error_rank_correlation"
                ),
                "high_error_region_capture": _mean_numeric(rows, "high_error_region_capture"),
                "near_neighbor_acquisition_rate": _mean_numeric(
                    rows, "near_neighbor_acquisition_rate"
                ),
                "runtime": float(
                    np.sum(
                        [
                            float(row["wall_time_seconds"])
                            for row in rows
                            if row.get("wall_time_seconds") not in (None, "")
                        ]
                    )
                ),
                "reselection_count": int(
                    sum(_as_bool(row.get("reselection_triggered")) for row in rows)
                ),
                "accepted_switch_count": int(
                    sum(
                        _as_bool(row.get("switch_accepted"))
                        and row.get("previous_kernel_id")
                        not in (None, "", row.get("selected_kernel_id"))
                        for row in rows
                    )
                ),
                "fraction_near_boundary": float(np.mean(boundary)) if boundary else None,
                "fraction_hull_vertices": float(np.mean(hull)) if hull else None,
                "median_nearest_observation_distance": (
                    float(np.median(distances)) if distances else None
                ),
                "fraction_selections_within_0_05_normalized_distance": (
                    float(np.mean(np.asarray(distances) <= 0.05)) if distances else None
                ),
                "fraction_selections_kernel_correlation_above_0_95": (
                    float(np.mean(np.asarray(correlations) > 0.95)) if correlations else None
                ),
            }
        )
    return summaries


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) not in (None, "") and np.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


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


def _state_touches_scale_bound(state: Any) -> bool:
    result = getattr(state, "kernel_selection_result", None)
    if result is None:
        return False
    values = np.asarray(state.current_length_scales, dtype=float)
    lower = np.asarray(result.optimization_event.length_scale_minimums, dtype=float)
    upper = np.asarray(result.optimization_event.length_scale_maximums, dtype=float)
    if values.size == 0 or values.shape != lower.shape or values.shape != upper.shape:
        return False
    tolerance = 1.0e-6
    return bool(np.any((values <= lower + tolerance) | (values >= upper - tolerance)))


def _normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept the compact nested YAML schema and retain legacy flat configs."""

    config = dict(raw or {})
    benchmark = dict(config.get("benchmark", {}))
    visualization = dict(config.get("visualization", {}))
    initial = config.get("initial_design", "interior_maximin")
    if isinstance(initial, dict):
        config["initial_design"] = initial.get("name", "interior_maximin")
        if "sample_count" in initial:
            config["initial_sample_count"] = initial["sample_count"]
        if "boundary_margin" in initial:
            config["initial_boundary_margin"] = initial["boundary_margin"]
    for key in (
        "trials",
        "initial_design_seeds",
        "final_budget",
        "candidate_count",
        "evaluation_grid_size",
        "base_seed",
        "initial_sample_count",
    ):
        if key in benchmark:
            config[key] = benchmark[key]
    for key, target in (
        ("save_gif", "save_gifs"),
        ("save_frames", "save_png_snapshots"),
        ("save_contact_sheet", "save_contact_sheet"),
        ("frame_duration_ms", "frame_duration_ms"),
        ("final_frame_duration_ms", "final_frame_duration_ms"),
        ("snapshot_every", "snapshot_every"),
        ("snapshot_sample_counts", "snapshot_sample_counts"),
        ("annotate_sample_order", "annotate_point_order"),
        ("dpi", "dpi"),
        ("save_pdf", "save_pdf"),
        ("optimize_hyperparameters", "optimize_hyperparameters"),
    ):
        if key in visualization:
            config[target] = visualization[key]
    candidate_validity = dict(config.get("candidate_validity", {}))
    if "minimum_physical_spacing" in benchmark:
        candidate_validity["minimum_normalized_distance"] = benchmark["minimum_physical_spacing"]
    config["candidate_validity"] = candidate_validity
    config.setdefault("methods", ["support_adjusted_krispu"])
    config.setdefault("initial_design", "interior_maximin")
    config.setdefault("initial_sample_count", 5)
    config.setdefault("initial_boundary_margin", 0.05)
    config.setdefault("base_seed", 202603)
    config.setdefault("save_contact_sheet", True)
    config.setdefault("final_frame_duration_ms", 1500)
    config.setdefault("save_png_snapshots", config.get("save_frames", False))
    config.setdefault("save_gifs", config.get("save_gif", False))
    return config


def _regular_grid(domain: Any, size: int) -> np.ndarray:
    axes = [np.linspace(lo, hi, size) for lo, hi in domain.bounds]
    mesh = np.meshgrid(*axes, indexing="xy")
    return np.column_stack([item.ravel() for item in mesh])


def _manifest(
    config: dict[str, Any], config_path: Path, seeds: list[dict[str, Any]]
) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    dependencies: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "sklearn", "matplotlib", "yaml"):
        try:
            module = __import__(name)
            dependencies[name] = getattr(module, "__version__", None)
        except ImportError:
            dependencies[name] = None
    return {
        "package_version": "0.3.0",
        "git_commit": commit,
        "python_version": platform.python_version(),
        "dependency_versions": dependencies,
        "configuration": config,
        "configuration_path": str(config_path),
        "seeds": seeds,
        "gpr_configuration": asdict(GPRConfig()),
        "field_definitions": {name: FIELD_FACTORIES[name]().metadata for name in config["fields"]},
        "candidate_pool": {"method": "lhs", "count": config["candidate_count"]},
        "evaluation_grid": {"size_per_axis": config["evaluation_grid_size"]},
        "jackknife_backend": "buffered_fixed_hyperparameters",
    }


if __name__ == "__main__":
    main()
