"""Command-line interface for one-shot KRISP-U recommendations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from krispu.recommendation import infer_continuous_space, recommend_next
from krispu.space import ContinuousSpace, DiscreteCandidateSpace, validate_bounds

MISSING_VALUES = {"", "na", "nan", "none", "null", "."}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    table = _read_csv(Path(args.data), delimiter=args.delimiter)
    feature_names = args.features or _infer_feature_columns(table.headers, args.target)
    observed_X, observed_y, embedded_candidates = _split_observed_and_candidates(
        table,
        feature_names=feature_names,
        target_name=args.target,
    )

    candidate_points = None
    if args.candidates is not None:
        candidate_table = _read_csv(Path(args.candidates), delimiter=args.delimiter)
        candidate_points = _read_feature_matrix(candidate_table, feature_names)
        space = DiscreteCandidateSpace(candidate_points, names=feature_names)
    elif len(embedded_candidates) > 0 and not args.ignore_empty_target_candidates:
        candidate_points = embedded_candidates
        space = DiscreteCandidateSpace(candidate_points, names=feature_names)
    else:
        bounds = (
            _parse_bounds(args.bounds, feature_names)
            if args.bounds
            else infer_continuous_space(
                observed_X,
                feature_names=feature_names,
                padding_fraction=args.padding_fraction,
            ).bounds
        )
        space = ContinuousSpace(bounds, names=feature_names)

    recommendations = recommend_next(
        observed_X,
        observed_y,
        space=space,
        n_recommendations=args.n_recommendations,
        objective=args.objective,
        acquisition=args.acquisition,
        candidates=candidate_points,
        n_candidates=args.n_candidates,
        candidate_method=args.candidate_method,
        random_state=args.random_state,
        feature_names=feature_names,
        exclude_observed=not args.allow_observed,
        optimize_continuous_acquisition=False,
    )

    output_path = Path(args.output)
    _write_recommendations(output_path, recommendations.to_records())
    print(
        f"Wrote {len(recommendations.recommendations)} recommendations to {output_path}"
    )
    print(f"Best observed {args.target}: {recommendations.best_observed_y:.6g}")
    print(f"Best observed point: {recommendations.best_observed_x}")
    if recommendations.selected_kernel_family is not None:
        print(f"Selected kernel: {recommendations.selected_kernel_family}")
    return 0


class CsvTable:
    """Small typed container for CSV rows."""

    def __init__(self, headers: list[str], rows: list[dict[str, str]]) -> None:
        self.headers = headers
        self.rows = rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="krispu-recommend",
        description="Fit KRISP-U to measured data and write ranked next-point recommendations.",
    )
    parser.add_argument("data", help="CSV containing measured rows.")
    parser.add_argument(
        "--features",
        nargs="+",
        help="Feature/input column names. Defaults to all columns except target.",
    )
    parser.add_argument(
        "--target", required=True, help="Measured response column name."
    )
    parser.add_argument(
        "--objective",
        choices=("minimize", "maximize"),
        default="minimize",
        help="Response direction for improvement-style acquisition functions.",
    )
    parser.add_argument(
        "--bounds",
        nargs="*",
        default=None,
        help=(
            "Continuous bounds as name:low:high entries, e.g. "
            "--bounds temp:20:80 pressure:1:5. If omitted, bounds are inferred."
        ),
    )
    parser.add_argument(
        "--candidates",
        help=(
            "Optional CSV of preset candidate rows. If omitted, rows in the main "
            "CSV with blank target values are treated as candidates."
        ),
    )
    parser.add_argument(
        "--ignore-empty-target-candidates",
        action="store_true",
        help="Ignore blank-target rows in the main CSV and use continuous candidates.",
    )
    parser.add_argument(
        "--n-recommendations",
        type=int,
        default=3,
        help="Number of ranked points to recommend.",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=4096,
        help="Generated candidate count for continuous recommendation.",
    )
    parser.add_argument(
        "--candidate-method",
        default="lhs",
        choices=("random", "lhs", "latin_hypercube", "sobol", "grid"),
        help="How to generate continuous candidate points.",
    )
    parser.add_argument(
        "--acquisition",
        default="uncertainty",
        help=(
            "Acquisition function. Defaults to uncertainty for field matching; "
            "kld uses field information gain, and expected_improvement is "
            "available for optimization-style runs."
        ),
    )
    parser.add_argument(
        "--padding-fraction",
        type=float,
        default=0.05,
        help="Range padding used when continuous bounds are inferred from observations.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--output", default="krispu_recommendations.csv")
    parser.add_argument(
        "--allow-observed",
        action="store_true",
        help="Allow already-measured points to appear in the recommendation pool.",
    )
    return parser


def _read_csv(path: Path, delimiter: str = ",") -> CsvTable:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        headers = [header.strip() for header in reader.fieldnames]
        rows = [
            {
                key.strip(): (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{path} contains no data rows.")
    return CsvTable(headers=headers, rows=rows)


def _infer_feature_columns(headers: list[str], target_name: str) -> list[str]:
    if target_name not in headers:
        raise ValueError(f"target column '{target_name}' was not found.")
    features = [header for header in headers if header != target_name]
    if not features:
        raise ValueError("At least one feature column is required.")
    return features


def _split_observed_and_candidates(
    table: CsvTable,
    feature_names: list[str],
    target_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    observed_X: list[list[float]] = []
    observed_y: list[float] = []
    candidate_X: list[list[float]] = []

    for row in table.rows:
        features = [_parse_float(row[name], name) for name in feature_names]
        target_value = row.get(target_name, "")
        if _is_missing(target_value):
            candidate_X.append(features)
            continue
        observed_X.append(features)
        observed_y.append(_parse_float(target_value, target_name))

    if len(observed_X) < 2:
        raise ValueError("At least two measured rows are required.")

    candidates = (
        np.asarray(candidate_X, dtype=float)
        if candidate_X
        else np.empty((0, len(feature_names)), dtype=float)
    )
    return (
        np.asarray(observed_X, dtype=float),
        np.asarray(observed_y, dtype=float),
        candidates,
    )


def _read_feature_matrix(
    table: CsvTable,
    feature_names: list[str],
) -> NDArray[np.float64]:
    rows = [
        [_parse_float(row[name], name) for name in feature_names] for row in table.rows
    ]
    if not rows:
        raise ValueError("Candidate file contains no rows.")
    return np.asarray(rows, dtype=float)


def _parse_bounds(
    bounds: list[str],
    feature_names: list[str],
) -> NDArray[np.float64]:
    parsed: dict[str, tuple[float, float]] = {}
    for item in _flatten_bound_items(bounds):
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError("Bounds must use name:low:high format, e.g. temp:20:80.")
        name, low, high = parts
        parsed[name] = (float(low), float(high))
    missing = [name for name in feature_names if name not in parsed]
    if missing:
        raise ValueError(f"Missing bounds for feature columns: {missing}")
    return validate_bounds([parsed[name] for name in feature_names])


def _flatten_bound_items(bounds: list[str]) -> list[str]:
    items: list[str] = []
    for bound in bounds:
        items.extend(part.strip() for part in bound.split(",") if part.strip())
    return items


def _write_recommendations(
    path: Path,
    rows: list[dict[str, float | int]],
) -> None:
    if not rows:
        raise ValueError("No recommendations to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_float(value: str, name: str) -> float:
    if _is_missing(value):
        raise ValueError(f"Missing numeric value for column '{name}'.")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Column '{name}' must contain numeric values.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"Column '{name}' must contain finite values.")
    return parsed


def _is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_VALUES


if __name__ == "__main__":
    raise SystemExit(main())
