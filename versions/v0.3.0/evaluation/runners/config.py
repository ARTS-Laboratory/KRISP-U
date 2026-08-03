"""Validated YAML profiles and safe overwrite-only evaluation outputs."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evaluation.fields import FIELD_FACTORIES
from krispu.config import BufferedJackknifeConfig

_TOP_LEVEL_KEYS = {
    "study",
    "experiment_name",
    "fields",
    "methods",
    "include_noisy_field",
    "initial_design",
    "initial_sample_count",
    "initial_boundary_margin",
    "candidate_validity",
    "minimum_normalized_distance",
    "final_budget",
    "candidate_count",
    "evaluation_grid_size",
    "trials",
    "base_seed",
    "initial_design_seeds",
    "save_gifs",
    "save_png_snapshots",
    "save_snapshot_gifs",
    "snapshot_every",
    "snapshot_sample_counts",
    "frame_duration_ms",
    "final_frame_duration_ms",
    "dpi",
    "annotate_point_order",
    "annotate_sample_order",
    "save_point_layout_animations",
    "save_comparison_figures",
    "save_pdf",
    "optimize_hyperparameters",
    "save_gif",
    "save_frames",
    "save_contact_sheet",
    "kernel_selection",
    "kernel",
    "jackknife",
    "visual_audit_fields",
    "visual_audit_modes",
    "manual",
    "noise_seed",
    "output",
    "benchmark",
    "visualization",
}
_NESTED_KEYS = {
    "candidate_validity": {"minimum_normalized_distance"},
    "output": {"mode"},
    "initial_design": {"name", "sample_count", "boundary_margin"},
    "benchmark": {
        "trials",
        "initial_design_seeds",
        "final_budget",
        "candidate_count",
        "evaluation_grid_size",
        "base_seed",
        "minimum_physical_spacing",
    },
    "visualization": {
        "save_gif",
        "save_frames",
        "save_contact_sheet",
        "frame_duration_ms",
        "final_frame_duration_ms",
        "snapshot_every",
        "snapshot_sample_counts",
        "annotate_sample_order",
        "dpi",
        "save_pdf",
        "optimize_hyperparameters",
    },
    "kernel_selection": {
        "enabled", "candidate_set", "profile",
        "mode",
        "optimization",
        "reselection",
        "optimizer_restarts",
        "reevaluate_every",
        "minimum_points_before_selection",
        "minimum_score_improvement",
        "random_state",
    },
    "kernel": {
        "enabled", "mode", "candidate_set", "profile", "specification", "optimization", "reselection",
        "random_state", "optimize_hyperparameters",
    },
    "jackknife": {"buffer"},
    "manual": {"optimizer_restarts", "optimize_hyperparameters"},
}


def load_config(path: Path) -> dict[str, Any]:
    """Load, validate, and resolve one YAML profile."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("The YAML profile must contain a mapping at its root.")
    _reject_unknown_keys(raw)
    config = _normalize_config(raw)
    _validate_config(config)
    return config


def write_resolved_config(output: Path, config: dict[str, Any]) -> None:
    """Persist the exact normalized settings used for a run."""

    output.joinpath("config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


def prepare_suite_output(output_root: Path, suite_name: str) -> Path:
    """Replace exactly one suite directory after checking deletion hazards."""

    root = Path(output_root).resolve()
    cwd = Path.cwd().resolve()
    if root == cwd or root.parent == root:
        raise ValueError("output_root must be a dedicated directory")
    root.mkdir(parents=True, exist_ok=True)
    output = root / suite_name
    if output.is_symlink():
        output.unlink()
    elif output.exists():
        if not output.is_dir():
            output.unlink()
        else:
            shutil.rmtree(output, onerror=_remove_readonly)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _remove_readonly(function: Any, path: str, _exc_info: Any) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _reject_unknown_keys(raw: dict[str, Any]) -> None:
    unknown = set(raw).difference(_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"Unknown YAML keys: {sorted(unknown)}")
    for section, allowed in _NESTED_KEYS.items():
        value = raw.get(section)
        if isinstance(value, dict):
            extra = set(value).difference(allowed)
            if extra:
                raise ValueError(f"Unknown YAML keys in {section}: {sorted(extra)}")


def _normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    config = dict(raw)
    if "kernel" in config and "kernel_selection" not in config:
        config["kernel_selection"] = config["kernel"]
    benchmark = dict(config.get("benchmark", {}))
    visualization = dict(config.get("visualization", {}))
    initial = config.get("initial_design", "interior_maximin")
    if isinstance(initial, dict):
        config["initial_design"] = initial.get("name", "interior_maximin")
        config.setdefault("initial_sample_count", initial.get("sample_count", 5))
        config.setdefault("initial_boundary_margin", initial.get("boundary_margin", 0.05))
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
    config.setdefault(
        "jackknife",
        {"buffer": {"mode": "median_nearest_neighbor", "multiplier": 1.0, "minimum_radius": 0.025, "maximum_radius": 0.20, "minimum_training_points": 3}},
    )
    config.setdefault("initial_design", "interior_maximin")
    config.setdefault("initial_sample_count", 5)
    config.setdefault("initial_boundary_margin", 0.05)
    config.setdefault("base_seed", 202603)
    config.setdefault("save_contact_sheet", False)
    config.setdefault("final_frame_duration_ms", 1500)
    config.setdefault("save_png_snapshots", False)
    config.setdefault("save_gifs", False)
    config.setdefault("output", {"mode": "summary"})
    if not isinstance(config["output"], dict):
        raise TypeError("output must be a mapping with mode summary, diagnostic, or debug.")
    config["output"] = {"mode": config["output"].get("mode", "summary")}
    return config


def jackknife_config(config: dict[str, Any]) -> BufferedJackknifeConfig:
    value = config.get("jackknife", {}).get("buffer", {})
    return BufferedJackknifeConfig(**value)


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
    if config["output"]["mode"] not in {"summary", "diagnostic", "debug"}:
        raise ValueError("output.mode must be summary, diagnostic, or debug.")
    unknown_fields = set(config["fields"]).difference(FIELD_FACTORIES)
    if unknown_fields:
        raise ValueError(f"Unknown fields: {sorted(unknown_fields)}")
    methods = {
        "support_adjusted_krispu",
        "raw_jackknife_sensitivity",
        "posterior_std",
        "random",
        "lhs",
        "maximin",
        "krispu",
        "krispu_fixed_gaussian",
        "krispu_fixed_matern32",
        "krispu_manual",
        "krispu_adaptive",
        "gp_posterior_variance",
        "random_sequential",
    }
    unknown_methods = set(config["methods"]).difference(methods)
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
    if minimum_normalized_distance(config) < 0:
        raise ValueError("minimum_normalized_distance must be non-negative.")
    if int(config["final_budget"]) < int(config["initial_sample_count"]):
        raise ValueError("final_budget must be at least initial_sample_count.")
    if config.get("snapshot_every") is not None and int(config["snapshot_every"]) <= 0:
        raise ValueError("snapshot_every must be positive when provided.")
    if int(config.get("frame_duration_ms", 500)) <= 0 or int(config.get("dpi", 150)) <= 0:
        raise ValueError("frame_duration_ms and dpi must be positive.")


def minimum_normalized_distance(config: dict[str, Any]) -> float:
    value = config.get("candidate_validity", {}).get(
        "minimum_normalized_distance",
        config.get("minimum_normalized_distance", 1.0e-4),
    )
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError("minimum_normalized_distance must be finite and non-negative.")
    return result


__all__ = [
    "load_config",
    "minimum_normalized_distance",
    "prepare_suite_output",
    "write_resolved_config",
]
