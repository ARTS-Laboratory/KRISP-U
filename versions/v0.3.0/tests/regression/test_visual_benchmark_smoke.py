import csv
from pathlib import Path

import yaml

from benchmarks.runner import run_benchmark


def test_visual_benchmark_smoke(tmp_path: Path) -> None:
    config = {
        "experiment_name": "test_smoke",
        "fields": ["smooth"],
        "methods": ["krispu_loo", "posterior_std", "random"],
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
    assert (output / "iteration_metrics.csv").exists()
    assert (output / "final_metrics.csv").exists()
    assert (output / "paired_comparisons.csv").exists()
    assert list((output / "figures").rglob("*.png"))
    with (output / "iteration_metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    assert metrics
    assert all(row["nrmse"] for row in metrics)
