"""CLI for reproducible v0.3.0 visual and paired performance audits."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from benchmarks.fields import FIELD_FACTORIES
from benchmarks.plotting import (
    plot_component_evolution,
    plot_error_concentration,
    plot_field_audit,
    plot_learning_curves,
    plot_paired_differences,
    plot_sampling_paths,
    plot_uncertainty_components,
    plot_uncertainty_error,
)
from benchmarks.records import RECORD_FIELDS, save_spatial_state, write_records
from krispu import GPRConfig
from krispu.candidates import generate_candidates
from krispu.sequential import run_sequential_design


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("benchmark_outputs"))
    args = parser.parse_args()
    output = run_benchmark(args.config, args.output_root)
    print(output)


def run_benchmark(config_path: Path, output_root: Path = Path("benchmark_outputs")) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_root / f"{timestamp}_{config['experiment_name']}"
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
        )
    }
    for directory in figure_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    (output / "spatial_arrays").mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    all_records: list[dict[str, Any]] = []
    final_records: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    seeds: list[dict[str, int | str]] = []
    final_by_field: dict[str, dict[str, Any]] = {}
    component_records: list[dict[str, Any]] = []
    for field_index, field_name in enumerate(config["fields"]):
        field = FIELD_FACTORIES[field_name]()
        evaluation = _regular_grid(field.domain, int(config["evaluation_grid_size"]))
        field_final: dict[str, Any] = {}
        for trial in range(int(config["trials"])):
            trial_seed = int(config["base_seed"]) + field_index * 100_000 + trial * 10_000
            field_seed = trial_seed + 1
            initial_design_seed = trial_seed + 2
            candidate_seed = trial_seed + 3
            candidate_pool = generate_candidates(
                field.domain, int(config["candidate_count"]), "lhs", candidate_seed
            )
            initial_X = _initial_design(config["initial_design"])
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
            for method_index, method in enumerate(config["methods"]):
                method_seed = trial_seed + 100 + method_index
                seeds[-1][f"{method}_method_seed"] = method_seed
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
                    gpr_config=GPRConfig(random_state=method_seed),
                )
                final_state = states[-1]
                field_final[method] = final_state
                for state in states:
                    record = state.scalar_record()
                    all_records.append(record)
                    if state.sample_count == int(config["final_budget"]):
                        final_records.append(record)
                    component_records.append(record)
                    array_name = f"{field_name}_trial{trial}_{method}_n{state.sample_count}.npz"
                    save_spatial_state(output / "spatial_arrays" / array_name, state)
                if method == "krispu_combined" and trial == 0:
                    snapshot_counts = {int(value) for value in config["snapshot_sample_counts"]}
                    for state in states:
                        if state.sample_count in snapshot_counts:
                            plot_field_audit(
                                state,
                                figure_dirs["field_audits"]
                                / f"{field_name}_n{state.sample_count}.png",
                                bool(config.get("save_pdf", False)),
                            )
                if method == "krispu_combined" and trial == 0:
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
            for baseline in ("krispu_jackknife", "posterior_std", "random", "lhs", "maximin"):
                if baseline in field_final and "krispu_combined" in field_final:
                    paired_rows.append(
                        {
                            "field": field_name,
                            "trial": trial,
                            "baseline": baseline,
                            "delta_nrmse": field_final["krispu_combined"].metrics.nrmse
                            - field_final[baseline].metrics.nrmse,
                            "krispu_nrmse": field_final["krispu_combined"].metrics.nrmse,
                            "baseline_nrmse": field_final[baseline].metrics.nrmse,
                        }
                    )
        final_by_field[field_name] = field_final
        plot_sampling_paths(
            field,
            field_final,
            figure_dirs["sampling_paths"] / f"{field_name}_sampling_paths.png",
            bool(config.get("save_pdf", False)),
        )
        field_rows = [row for row in component_records if row["field"] == field_name]
        plot_component_evolution(
            field_rows,
            figure_dirs["component_evolution"] / f"{field_name}_components.png",
            bool(config.get("save_pdf", False)),
        )
    plot_learning_curves(all_records, figure_dirs["learning_curves"], int(config["trials"]) > 1)
    plot_paired_differences(
        paired_rows, figure_dirs["paired_performance"], bool(config.get("save_pdf", False))
    )
    write_records(output / "iteration_metrics.csv", all_records, RECORD_FIELDS)
    write_records(output / "final_metrics.csv", final_records, RECORD_FIELDS)
    paired_fields = ["field", "trial", "baseline", "delta_nrmse", "krispu_nrmse", "baseline_nrmse"]
    write_records(output / "paired_comparisons.csv", paired_rows, paired_fields)
    manifest = _manifest(config, config_path, seeds)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "experiment_name",
        "fields",
        "methods",
        "initial_sample_count",
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
    unknown_methods = set(config["methods"]).difference(
        {"krispu_combined", "krispu_jackknife", "posterior_std", "random", "lhs", "maximin"}
    )
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")
    if config["initial_sample_count"] != 5:
        raise ValueError("The v0.3.0 audit uses the five-point corners_plus_center design.")
    if int(config["final_budget"]) < int(config["initial_sample_count"]):
        raise ValueError("final_budget must be at least initial_sample_count.")


def _initial_design(name: str) -> np.ndarray:
    if name != "corners_plus_center":
        raise ValueError("Only corners_plus_center is implemented in v0.3.0.")
    return np.asarray([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])


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
        "loo_backend": "brute_force",
    }


if __name__ == "__main__":
    main()
