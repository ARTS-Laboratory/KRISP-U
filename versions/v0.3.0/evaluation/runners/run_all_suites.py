"""Run the complete v0.3.0 benchmark suite in a reproducible order."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from sklearn.exceptions import ConvergenceWarning
from tqdm.auto import tqdm

from evaluation.runners.suite import run_benchmark

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUITE_NAMES: tuple[str, ...] = (
    "development",
    "kernel_recovery",
    "canonical_2d",
    "canonical_doe",
    "noise_robustness",
    "full_evaluation",
)


def run_all_suites(
    suite_names: Sequence[str] = SUITE_NAMES,
    output_root: Path | None = None,
    *,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    """Run selected suites and persist one status manifest.

    Each suite retains the existing overwrite-only output directory contract.
    The wrapper adds no timestamps and does not create a second run hierarchy.
    """

    names = tuple(suite_names)
    unknown = sorted(set(names).difference(SUITE_NAMES))
    if unknown:
        raise ValueError(f"Unknown suite names: {unknown}")
    root = (REPOSITORY_ROOT / "outputs") if output_root is None else Path(output_root)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    manifest_path = root / "all_suites_manifest.yaml"

    progress = tqdm(names, desc="KRISP-U suites", unit="suite")
    for name in progress:
        progress.set_postfix_str(name)
        config_path = REPOSITORY_ROOT / "configs" / "suites" / f"{name}.yaml"
        result: dict[str, Any] = {
            "suite": name,
            "config": str(config_path),
            "output": str(root / name),
            "status": "pending",
        }
        tqdm.write(f"[run-all] {name}: starting")
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ConvergenceWarning)
                output = run_benchmark(config_path, root)
            result["expected_convergence_warnings"] = sum(
                warning.category is ConvergenceWarning for warning in captured
            )
        except Exception as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
            result.setdefault("expected_convergence_warnings", 0)
            tqdm.write(f"[run-all] {name}: failed ({result['error']})")
            results.append(result)
            _write_manifest(manifest_path, names, results)
            if not continue_on_error:
                raise
            continue
        result["status"] = "completed"
        result["output"] = str(output)
        results.append(result)
        _write_manifest(manifest_path, names, results)
        tqdm.write(
            f"[run-all] {name}: completed -> {output} "
            f"(expected convergence warnings: {result['expected_convergence_warnings']})",
        )

    return {
        "suite_order": list(names),
        "results": results,
        "manifest": str(manifest_path),
    }


def _write_manifest(path: Path, suite_names: Sequence[str], results: list[dict[str, Any]]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "wrapper": "evaluation.runners.run_all_suites",
                "suite_order": list(suite_names),
                "results": results,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        choices=SUITE_NAMES,
        dest="suite_names",
        help="Run only this suite; repeat the option to select several suites.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory; defaults to versions/v0.3.0/outputs.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed suite.",
    )
    args = parser.parse_args()
    summary = run_all_suites(
        args.suite_names or SUITE_NAMES,
        args.output_root,
        continue_on_error=not args.fail_fast,
    )
    return int(any(result["status"] == "failed" for result in summary["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
