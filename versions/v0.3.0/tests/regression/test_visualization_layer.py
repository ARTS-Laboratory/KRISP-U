from pathlib import Path

import numpy as np
import yaml

from evaluation.fields.canonical.noisy import noisy_field
from evaluation.figures.sequential import snapshot_sample_counts
from evaluation.runners.suite import run_benchmark


def test_noisy_field_is_deterministic_from_seed() -> None:
    points = np.array([[-1.0, -1.0], [-0.25, 0.4], [1.0, 1.0]])
    first = noisy_field(17).evaluate(points)
    second = noisy_field(17).evaluate(points)
    different = noisy_field(18).evaluate(points)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def test_snapshot_schedule_prioritizes_explicit_counts() -> None:
    assert snapshot_sample_counts(12, 5, snapshot_every=2, snapshot_sample_counts=[7, 11]) == (
        5,
        7,
        11,
        12,
    )
    assert snapshot_sample_counts(12, 5, snapshot_every=3) == (5, 8, 11, 12)


def test_noisy_visual_audit_writes_outputs_under_requested_root(tmp_path: Path) -> None:
    config = {
        "experiment_name": "noisy_visual_test",
        "include_noisy_field": True,
        "fields": ["noisy_baseline"],
        "methods": ["support_adjusted_krispu", "posterior_std", "random"],
        "initial_design": "interior_maximin",
        "initial_sample_count": 5,
        "initial_boundary_margin": 0.05,
        "minimum_normalized_distance": 0.05,
        "final_budget": 6,
        "candidate_count": 24,
        "evaluation_grid_size": 10,
        "trials": 1,
        "base_seed": 21,
        "snapshot_sample_counts": [5, 6],
        "save_gifs": True,
        "save_png_snapshots": True,
        "save_point_layout_animations": True,
        "save_comparison_figures": True,
        "frame_duration_ms": 80,
        "dpi": 80,
        "output": {"mode": "debug"},
    }
    config_path = tmp_path / "noisy.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_root = tmp_path / "benchmark_outputs"
    output = run_benchmark(config_path, output_root)

    assert output.resolve().is_relative_to(output_root.resolve())
    assert list((output / "animations").rglob("*.gif"))
    assert list((output / "snapshots").rglob("*.png"))
    assert (output / "report.md").exists()
    assert (output / "config_resolved.yaml").exists()
