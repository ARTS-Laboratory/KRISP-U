from __future__ import annotations

import csv

import numpy as np

from krispu import ContinuousSpace, DiscreteCandidateSpace, recommend_next
from krispu.cli import main
from krispu.datasets import get_dataset
from krispu.models import GprConfig


def test_recommend_next_ranks_discrete_candidates() -> None:
    dataset = get_dataset("branin")
    observed_X = dataset.initial_design(n=5, random_state=12)
    observed_y = dataset.evaluate(observed_X)
    candidates = np.asarray(
        [
            [-np.pi, 12.275],
            [3.1416, 2.275],
            [9.4248, 2.475],
            [0.0, 8.0],
        ]
    )
    space = DiscreteCandidateSpace(candidates, names=("x1", "x2"))
    result = recommend_next(
        observed_X,
        observed_y,
        space=space,
        n_recommendations=2,
        objective=dataset.objective,
        candidates=candidates,
        random_state=12,
    )

    assert len(result.recommendations) == 2
    assert result.acquisition == "uncertainty"
    assert result.feature_names == ["x1", "x2"]
    assert (
        result.recommendations[0].acquisition_score
        >= result.recommendations[1].acquisition_score
    )
    assert bool(space.contains(result.recommendations[0].x)[0])


def test_recommend_next_supports_continuous_spaces() -> None:
    dataset = get_dataset("quadratic_bowl_2d")
    observed_X = dataset.initial_design(n=5, random_state=13)
    observed_y = dataset.evaluate(observed_X)
    space = ContinuousSpace(dataset.bounds, names=dataset.labels)

    result = recommend_next(
        observed_X,
        observed_y,
        space=space,
        n_recommendations=3,
        n_candidates=32,
        random_state=13,
        gpr_config=GprConfig(n_restarts_optimizer=0),
    )

    assert len(result.recommendations) == 3
    assert np.all(space.contains(result.as_array()))


def test_cli_writes_recommendation_csv(tmp_path) -> None:
    data_path = tmp_path / "measurements.csv"
    output_path = tmp_path / "recommendations.csv"
    rows = [
        {"x1": "-4.0", "x2": "14.0", "response": "3.958293"},
        {"x1": "-2.0", "x2": "4.0", "response": "30.602112"},
        {"x1": "2.5", "x2": "12.0", "response": "24.129964"},
        {"x1": "5.0", "x2": "4.0", "response": "15.829732"},
        {"x1": "-3.1416", "x2": "12.275", "response": ""},
        {"x1": "3.1416", "x2": "2.275", "response": ""},
    ]
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x1", "x2", "response"])
        writer.writeheader()
        writer.writerows(rows)

    exit_code = main(
        [
            str(data_path),
            "--target",
            "response",
            "--features",
            "x1",
            "x2",
            "--n-recommendations",
            "2",
            "--output",
            str(output_path),
            "--random-state",
            "14",
        ]
    )

    assert exit_code == 0
    with output_path.open(newline="", encoding="utf-8") as handle:
        output_rows = list(csv.DictReader(handle))
    assert len(output_rows) == 2
    assert set(output_rows[0]) == {
        "rank",
        "acquisition_score",
        "predicted_mean",
        "predicted_std",
        "x1",
        "x2",
    }
