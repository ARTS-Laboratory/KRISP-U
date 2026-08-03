import csv
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from evaluation.fields import FIELD_FACTORIES
from evaluation.figures.sequential import panel_figure
from evaluation.runners.suite import run_benchmark


def _config(field: str, method: str, budget: int = 7) -> dict[str, object]:
    return {
        "experiment_name": "compact_test",
        "fields": [field],
        "methods": [method],
        "initial_design": "interior_maximin",
        "initial_sample_count": 5,
        "initial_boundary_margin": 0.05,
        "minimum_normalized_distance": 0.01,
        "final_budget": budget,
        "candidate_count": 48,
        "evaluation_grid_size": 10,
        "trials": 1,
        "base_seed": 123,
        "kernel_selection": {
            "mode": "automatic",
            "optimization": {"restarts": 0},
            "reselection": {"minimum_points": 6, "maximum_interval": 1},
        },
        "output": {"mode": "summary"},
    }


def test_field_registry_separates_true_gp_metadata_from_deterministic_fields() -> None:
    sampled = FIELD_FACTORIES["gp_rotated_anisotropic"](17)
    deterministic = FIELD_FACTORIES["Franke"]()
    assert sampled.metadata["true_kernel"]["family"] == "matern_32_ard"
    assert sampled.metadata["true_kernel"]["seed"] == 17
    assert deterministic.metadata["true_kernel"] is None


def test_adaptive_summary_tree_events_and_titles(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config("smooth", "krispu_adaptive")), encoding="utf-8")
    output = run_benchmark(config_path, tmp_path / "outputs")
    expected = {
        "config_resolved.yaml",
        "manifest.yaml",
        "report.md",
        "metrics/per_step.csv",
        "metrics/final.csv",
        "metrics/aggregate.csv",
        "kernel/events.csv",
        "kernel/candidate_scores.csv",
        "figures/fields/smooth/process.gif",
        "figures/fields/smooth/checkpoints.png",
        "figures/fields/smooth/learning_curve.png",
        "figures/fields/smooth/kernel_history.png",
        "figures/global/aggregate_learning_curve.png",
        "figures/global/performance_profile.png",
        "figures/global/kernel_ablation.png",
        "figures/global/robustness_matrix.png",
    }
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    assert actual == expected
    assert not (output / "spatial_arrays").exists()
    with (output / "kernel/events.csv").open(newline="", encoding="utf-8") as handle:
        events = list(csv.DictReader(handle))
    with (output / "kernel/candidate_scores.csv").open(newline="", encoding="utf-8") as handle:
        scores = list(csv.DictReader(handle))
    reselection_counts = {row["sample_count"] for row in events if row["reselection_triggered"] == "True"}
    assert {row["sample_count"] for row in scores} <= reselection_counts
    assert events
    assert all(row["optimization_completed"] for row in events)
    with Image.open(output / "figures/fields/smooth/process.gif") as animation:
        assert animation.n_frames == 3


def test_doe_summary_does_not_make_two_dimensional_field_plots(tmp_path: Path) -> None:
    config = _config("Hartmann 3D", "random_sequential", budget=6)
    config["candidate_count"] = 24
    config["evaluation_grid_size"] = 4
    config_path = tmp_path / "doe.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = run_benchmark(config_path, tmp_path / "outputs")
    assert not (output / "figures/fields/Hartmann 3D").exists()
    assert len(list((output / "figures/global").glob("*.png"))) == 4


def test_panel_title_contains_family_and_scales() -> None:
    field = FIELD_FACTORIES["smooth"]()
    points = np.asarray([[x, y] for x in np.linspace(-1, 1, 5) for y in np.linspace(-1, 1, 5)])
    from evaluation.runners.design import initial_design
    from evaluation.runners.sequential import run_sequential_design

    initial = initial_design("interior_maximin", field.domain, 5, 0.05, 9)
    states = run_sequential_design(
        field.evaluate,
        field.domain,
        initial,
        points,
        points,
        "posterior_std",
        5,
        11,
    )
    figure = panel_figure(states[-1])
    assert "family=" in figure._suptitle.get_text()
    assert "ARD=[" in figure._suptitle.get_text()
