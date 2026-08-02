"""Researcher-facing CSV recommendation command."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from krispu import ContinuousDomain, KrispURecommender, ObservationSet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recommend field measurements with KRISP-U v0.3.0."
    )
    parser.add_argument("measurements")
    parser.add_argument("--target", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument(
        "--bounds", nargs="+", required=True, help="Feature bounds such as x:-1:1 y:0:2"
    )
    parser.add_argument("--n-recommendations", type=int, default=1)
    parser.add_argument("--n-candidates", type=int, default=2048)
    parser.add_argument("--min-normalized-distance", type=float, default=1.0e-4)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    bounds = _parse_bounds(args.bounds, args.features)
    measured, candidate_rows = _read_csv(Path(args.measurements), args.features, args.target)
    if len(measured[0]) < 2:
        raise ValueError("At least two rows with finite target responses are required.")
    observations = ObservationSet(
        np.asarray(measured[0], dtype=float), np.asarray(measured[1], dtype=float)
    )
    domain = ContinuousDomain(bounds=bounds, names=tuple(args.features))
    recommender = KrispURecommender(
        domain,
        random_state=args.random_state,
        n_candidates=args.n_candidates,
        min_normalized_distance=args.min_normalized_distance,
    )
    result = recommender.recommend(
        observations, args.n_recommendations, candidates=candidate_rows or None
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = result.to_records()
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "rank",
            *args.features,
            "loo_field_sensitivity_at_selection",
            "kernel_support_deficit_at_selection",
            "krispu_uncertainty_at_selection",
            "predicted_mean",
            "posterior_std",
            "nearest_normalized_distance",
            "maximum_kernel_correlation_to_observations",
            "acquisition_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})
    return 0


def _parse_bounds(raw: list[str], features: list[str]) -> list[list[float]]:
    parsed: dict[str, list[float]] = {}
    for item in raw:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid bound {item!r}; use name:lower:upper.")
        name, lower, upper = parts
        parsed[name] = [float(lower), float(upper)]
    if set(parsed) != set(features):
        raise ValueError("--bounds must provide exactly one bound for each --features name.")
    return [parsed[name] for name in features]


def _read_csv(
    path: Path, features: list[str], target: str
) -> tuple[tuple[list[list[float]], list[float]], list[list[float]]]:
    measured_X: list[list[float]] = []
    measured_y: list[float] = []
    candidate_rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(
            name not in reader.fieldnames for name in [*features, target]
        ):
            raise ValueError("CSV is missing a requested feature or target column.")
        for row in reader:
            coordinates = [float(row[name]) for name in features]
            if row[target] is None or row[target].strip() == "":
                candidate_rows.append(coordinates)
            else:
                measured_X.append(coordinates)
                measured_y.append(float(row[target]))
    return (measured_X, measured_y), candidate_rows


if __name__ == "__main__":
    raise SystemExit(main())
