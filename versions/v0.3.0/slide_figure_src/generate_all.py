"""Generate the complete, intentionally scoped v0.3.0 slide-figure set."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .comparison_figures import build_comparison_runs, generate_summary_figures
from .doe_figures import generate_doe_figures
from .styles import configure
from .theory_figures import generate_theory_figures
from .toy_figures import build_toy_runs, write_toy_figures
from .workflow_figures import generate_workflow_figure


@dataclass(frozen=True)
class FigureConfig:
    """All stochastic and resolution settings needed to reproduce the figures."""

    seed: int = 202603
    dpi: int = 180
    final_budget: int = 20
    grid_size: int = 48
    candidate_count: int = 280
    animation_points: int = 36
    animation_frame_ms: int = 450
    animation_final_hold_ms: int = 2500
    jackknife_sample_count: int = 10
    jackknife_frame_ms: int = 650
    jackknife_final_hold_ms: int = 2500


GROUPS = (
    "doe_comparison",
    "workflow",
    "theory",
    "toy_smooth",
    "toy_rough_multiscale",
    "summary",
)


def generate_all(
    output_root: Path | None = None,
    *,
    seed: int = 202603,
    dpi: int = 180,
    final_budget: int = 20,
    grid_size: int = 48,
    candidate_count: int = 280,
    animation_points: int = 36,
    animation_frame_ms: int = 450,
    animation_final_hold_ms: int = 2500,
    jackknife_sample_count: int = 10,
    jackknife_frame_ms: int = 650,
    jackknife_final_hold_ms: int = 2500,
) -> dict[str, list[Path]]:
    """Generate only the five required slide topics and their summary figures."""

    configure()
    root = output_root or Path(__file__).resolve().parents[1] / "slide_figures"
    config = FigureConfig(
        seed,
        dpi,
        final_budget,
        grid_size,
        candidate_count,
        animation_points,
        animation_frame_ms,
        animation_final_hold_ms,
        jackknife_sample_count,
        jackknife_frame_ms,
        jackknife_final_hold_ms,
    )
    directories = {name: root / name for name in GROUPS}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    _write_config(root, config)

    toy_runs = build_toy_runs(
        config.seed,
        final_budget=config.final_budget,
        grid_size=config.grid_size,
        candidate_count=config.candidate_count,
    )
    outputs: dict[str, list[Path]] = {}
    outputs["doe_comparison"] = list(
        generate_doe_figures(
            directories["doe_comparison"],
            seed=config.seed,
            n_points=config.final_budget,
            krispu_points=toy_runs["smooth"].final_state.observed_X,
            animation_points=config.animation_points,
            animation_frame_ms=config.animation_frame_ms,
            animation_final_hold_ms=config.animation_final_hold_ms,
            dpi=config.dpi,
        ).values()
    )
    outputs["workflow"] = [
        generate_workflow_figure(directories["workflow"], dpi=config.dpi)
    ]
    outputs["theory"] = list(
        generate_theory_figures(
            directories["theory"],
            state=toy_runs["smooth"].final_state,
            run=toy_runs["smooth"],
            seed=config.seed,
            jackknife_sample_count=config.jackknife_sample_count,
            jackknife_frame_ms=config.jackknife_frame_ms,
            jackknife_final_hold_ms=config.jackknife_final_hold_ms,
            dpi=config.dpi,
        ).values()
    )
    outputs["toy_smooth"] = list(
        write_toy_figures(toy_runs["smooth"], directories["toy_smooth"], dpi=config.dpi).values()
    )
    outputs["toy_rough_multiscale"] = list(
        write_toy_figures(
            toy_runs["rough_multiscale"],
            directories["toy_rough_multiscale"],
            dpi=config.dpi,
        ).values()
    )

    comparison_runs = build_comparison_runs(
        config.seed + 1000,
        final_budget=config.final_budget,
        grid_size=config.grid_size,
        candidate_count=config.candidate_count,
        principal_runs=toy_runs,
    )
    outputs["summary"] = list(
        generate_summary_figures(
            directories["summary"],
            comparison_runs,
            data_root=root.parent / "outputs",
            dpi=config.dpi,
        ).values()
    )
    return outputs


def _write_config(root: Path, config: FigureConfig) -> None:
    payload: dict[str, Any] = {
        "version": "v0.3.0",
        "seed_and_resolution": asdict(config),
        "output_groups": list(GROUPS),
        "source_data": [
            "evaluation.fields.canonical.smooth",
            "evaluation.fields.synthetic_gp.kernel_fields.rough_multiscale_field",
            "evaluation.runners.sequential.run_sequential_design",
        ],
        "scope": {
            "required": [
                "existing_doe_approaches",
                "krispu_workflow",
                "krispu_theory",
                "smooth_toy_problem",
                "rough_multiscale_toy_problem",
            ],
            "excluded": [
                "additive_manufacturing",
                "air_force_application",
                "real_world_application_slides",
            ],
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "generation_config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=202603)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--final-budget", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--candidate-count", type=int, default=280)
    parser.add_argument("--animation-points", type=int, default=36)
    parser.add_argument("--animation-frame-ms", type=int, default=450)
    parser.add_argument("--animation-final-hold-ms", type=int, default=2500)
    parser.add_argument("--jackknife-sample-count", type=int, default=10)
    parser.add_argument("--jackknife-frame-ms", type=int, default=650)
    parser.add_argument("--jackknife-final-hold-ms", type=int, default=2500)
    args = parser.parse_args()
    generate_all(
        args.output_root,
        seed=args.seed,
        dpi=args.dpi,
        final_budget=args.final_budget,
        grid_size=args.grid_size,
        candidate_count=args.candidate_count,
        animation_points=args.animation_points,
        animation_frame_ms=args.animation_frame_ms,
        animation_final_hold_ms=args.animation_final_hold_ms,
        jackknife_sample_count=args.jackknife_sample_count,
        jackknife_frame_ms=args.jackknife_frame_ms,
        jackknife_final_hold_ms=args.jackknife_final_hold_ms,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["GROUPS", "FigureConfig", "generate_all"]
