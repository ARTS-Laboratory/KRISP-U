import csv
from pathlib import Path

import yaml

from evaluation.runners.suite import run_benchmark


def test_visual_benchmark_smoke(tmp_path: Path) -> None:
    config = {
        "experiment_name": "test_smoke",
        "fields": ["smooth"],
        "methods": ["support_adjusted_krispu", "posterior_std", "random"],
        "initial_design": "interior_maximin",
        "initial_sample_count": 5,
        "initial_boundary_margin": 0.05,
        "minimum_normalized_distance": 0.05,
        "final_budget": 7,
        "candidate_count": 32,
        "evaluation_grid_size": 12,
        "trials": 1,
        "base_seed": 8,
        "snapshot_sample_counts": [5, 7],
        "save_pdf": False,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    root = tmp_path / "v0.3.0-local"
    output = run_benchmark(config_path, root)
    assert output.parent == root
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
    with (output / "metrics/per_step.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    assert metrics
    assert all(row["nrmse"] for row in metrics)
    rerun = run_benchmark(config_path, root)
    assert rerun == output
    assert not (root / "test_smoke_2").exists()
