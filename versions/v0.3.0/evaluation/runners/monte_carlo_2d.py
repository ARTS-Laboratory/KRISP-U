"""Focused paired 2D Monte Carlo evaluation for KRISP-U v0.3.0."""

from __future__ import annotations

import csv
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: I001
import numpy as np
from PIL import Image
from sklearn.base import clone

from evaluation.metrics.reconstruction import reconstruction_metrics
from evaluation.runners.config import (
    jackknife_config,
    load_config,
    minimum_normalized_distance,
    prepare_suite_output,
    write_resolved_config,
)
from evaluation.runners.design import initial_design
from evaluation.fields.synthetic_gp.sampled import sampled_gp_field
from krispu.candidates import generate_candidates, valid_candidate_mask
from krispu.config import GPRConfig
from krispu.domains import ContinuousDomain
from krispu.jackknife import jackknife_field_sensitivity
from krispu.jackknife.plan import BufferedJackknifePlan, build_buffered_jackknife_plan
from krispu.kernels.builders import build_named_kernel
from krispu.surrogates.gpr import GPRSurrogate
from krispu.uncertainty.support import kernel_support_deficit


FIELD_NAMES = (
    "smooth_single_scale",
    "clustered_observations",
    "branin",
    "baseline_drift",
    "white_noise",
    "heteroscedastic_noise",
    "matched_spherical",
)
METHOD_NAMES = (
    "adaptive_krispu",
    "adaptive_kriging",
    "raw_jackknife_sensitivity",
    "progressive_lhs",
    "sequential_maximin",
    "random",
)
ADAPTIVE_METHOD_NAMES = (
    "adaptive_krispu",
    "adaptive_kriging",
    "raw_jackknife_sensitivity",
)
BASELINE_METHOD_NAMES = ("progressive_lhs", "sequential_maximin", "random")
KERNEL_FAMILIES = (
    "gaussian_ard",
    "exponential_ard",
    "spherical_ard",
    "matern_32_ard",
    "matern_52_ard",
    "rational_quadratic_ard",
    "wendland_c2_ard",
)
OUTPUT_NAME = "monte_carlo_2d"
PER_STEP_FIELDS = (
    "field",
    "trial",
    "method",
    "sample_count",
    "r2",
    "nrmse",
    "runtime",
    "selected_kernel",
    "length_scale_x",
    "length_scale_y",
    "reselection_triggered",
    "switch_accepted",
)
TRIAL_FIELDS = (
    "field",
    "trial",
    "method",
    "final_r2",
    "r2_auc",
    "final_nrmse",
    "runtime",
    "reselection_count",
    "switch_count",
)
PAIRED_FIELDS = (
    "field",
    "trial",
    "adaptive_method",
    "baseline",
    "metric",
    "adaptive_value",
    "baseline_value",
    "adaptive_minus_baseline",
)
EVENT_FIELDS = (
    "field",
    "trial",
    "method",
    "sample_count",
    "event_type",
    "kernel_family",
    "previous_family",
    "selected_family",
    "family_check",
    "check_retained",
    "reselection_triggered",
    "switch_accepted",
    "trigger_reasons",
    "ranked_families",
    "validated_families",
    "log_marginal_likelihood",
    "validation_score",
    "challenger_validation_score",
    "score_improvement",
    "length_scale_x",
    "length_scale_y",
    "optimization_runtime",
    "family_fit_count",
)


@dataclass(frozen=True)
class TrialField:
    name: str
    domain: ContinuousDomain
    clean: Callable[[np.ndarray], np.ndarray]
    observed: Callable[[np.ndarray], np.ndarray]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Scenario:
    field: TrialField
    trial: int
    seed: int
    initial_X: np.ndarray
    candidate_pool: np.ndarray
    evaluation_points: np.ndarray
    evaluation_truth: np.ndarray
    visualization_points: np.ndarray
    visualization_truth: np.ndarray


@dataclass(frozen=True)
class SelectionOutcome:
    surrogate: GPRSurrogate
    family: str
    validation_score: float
    reselection_triggered: bool
    switch_accepted: bool
    check_retained: bool
    trigger_reasons: tuple[str, ...]
    previous_family: str | None
    optimization_runtime: float
    event_row: dict[str, Any]


@dataclass(frozen=True)
class TrajectoryState:
    truth: np.ndarray
    prediction: np.ndarray
    uncertainty: np.ndarray
    map_points: np.ndarray
    observed_X: np.ndarray
    newest_point: np.ndarray
    field: str
    sample_count: int
    r2: float
    method: str = "adaptive_krispu"


class AdaptiveKernelController:
    """One global ARD family with triggered five-family checks."""

    def __init__(
        self,
        random_state: int,
        base_config: GPRConfig,
        *,
        initial_family: str = "matern_32_ard",
        fixed_family: str | None = None,
    ) -> None:
        if initial_family not in KERNEL_FAMILIES:
            raise ValueError(f"Unsupported initial kernel family: {initial_family}")
        if fixed_family is not None and fixed_family not in KERNEL_FAMILIES:
            raise ValueError(f"Unsupported fixed kernel family: {fixed_family}")
        self.random_state = random_state
        self.base_config = base_config
        self.family = fixed_family or initial_family
        self.fixed_family = fixed_family
        self.kernel: Any | None = None
        self.last_validation: float | None = None
        self.last_check_count: int | None = None
        self.bound_contact_steps = 0
        self.has_checked = False
        self.optimization_count = 0
        self.family_check_count = 0
        self.switch_count = 0

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_count: int,
    ) -> SelectionOutcome:
        normalized = X
        plan = _jackknife_plan(normalized, self.base_config)
        started = perf_counter()
        current_fit: GPRSurrogate | None = None
        fit_failure = False
        try:
            current_fit = self._fit_family(self.family, normalized, y, self.kernel)
        except (FloatingPointError, np.linalg.LinAlgError, RuntimeError, ValueError):
            fit_failure = True

        current_score = (
            np.inf
            if current_fit is None or plan is None
            else _buffered_validation_score(current_fit, normalized, y, plan)
        )
        reasons: list[str] = []
        if plan is not None and not self.has_checked:
            reasons.append("first eligible selection")
        if fit_failure:
            reasons.append("fit failure")
        if (
            self.last_validation is not None
            and np.isfinite(current_score)
            and current_score
            > self.last_validation + max(0.10 * abs(self.last_validation), 0.01)
        ):
            reasons.append("validation worsened >10%")
        near_bound = (
            current_fit is not None and _length_scale_near_bound(current_fit.frozen_kernel)
        )
        self.bound_contact_steps = self.bound_contact_steps + 1 if near_bound else 0
        if plan is not None and self.bound_contact_steps >= 2:
            reasons.append("scale near bound for two steps")
        if (
            plan is not None
            and self.last_check_count is not None
            and sample_count - self.last_check_count >= 5
        ):
            reasons.append("five points since previous check")

        triggered = bool(self.fixed_family is None and plan is not None and reasons)
        previous = self.family
        selected = current_fit
        selected_family = self.family
        selected_score = current_score
        accepted = False
        retained = False
        challenger_score: float | None = None
        improvement = 0.0
        ranked: tuple[str, ...] = ()
        validated: tuple[str, ...] = ()
        optimization_runtime = perf_counter() - started
        if triggered:
            check_started = perf_counter()
            fits: dict[str, GPRSurrogate] = {}
            log_likelihoods: dict[str, float] = {}
            for family in KERNEL_FAMILIES:
                try:
                    warm = self.kernel if family == self.family else None
                    fit = self._fit_family(family, normalized, y, warm)
                    fits[family] = fit
                    log_likelihoods[family] = fit.log_marginal_likelihood
                except (FloatingPointError, np.linalg.LinAlgError, RuntimeError, ValueError):
                    log_likelihoods[family] = -np.inf
            ranked = tuple(sorted(KERNEL_FAMILIES, key=lambda item: log_likelihoods[item], reverse=True))
            challengers = [family for family in ranked if family != self.family][:2]
            validated = tuple(dict.fromkeys([self.family, *challengers]))
            validation: dict[str, float] = {}
            for family in validated:
                fit = fits.get(family)
                validation[family] = (
                    np.inf
                    if fit is None or plan is None
                    else _buffered_validation_score(fit, normalized, y, plan)
                )
            current_score = validation.get(self.family, np.inf)
            best_challenger = min(
                (family for family in validated if family != self.family),
                key=lambda item: validation[item],
                default=None,
            )
            if best_challenger is not None:
                challenger_score = validation[best_challenger]
                if np.isfinite(current_score) and np.isfinite(challenger_score):
                    improvement = (
                        current_score - challenger_score
                    ) / max(abs(current_score), 1.0e-12)
                elif not np.isfinite(current_score) and np.isfinite(challenger_score):
                    improvement = np.inf
                accepted = bool(improvement >= 0.05)
                if accepted:
                    selected_family = best_challenger
                else:
                    retained = True
            else:
                retained = True
            selected = fits.get(selected_family)
            if selected is None:
                selected_family = self.family
                selected = current_fit
            if selected is None:
                raise RuntimeError("All adaptive kernel families failed to fit.")
            selected_score = validation.get(selected_family, np.inf)
            self.family_check_count += 1
            self.last_check_count = sample_count
            self.has_checked = True
            if accepted and selected_family != previous:
                self.switch_count += 1
            optimization_runtime = perf_counter() - check_started
        if selected is None:
            raise RuntimeError("Adaptive global-kernel fit failed before an eligible check.")

        if not triggered:
            selected_family = self.family
            selected_score = current_score
        switch = bool(accepted and selected_family != previous)
        self.family = selected_family
        self.kernel = selected.frozen_kernel
        self.last_validation = selected_score
        scales = _length_scales(selected.frozen_kernel)
        event_row = {
            "event_type": "family_check" if triggered else "optimization",
            "kernel_family": selected_family,
            "previous_family": previous,
            "selected_family": selected_family,
            "family_check": triggered,
            "check_retained": retained if triggered else False,
            "reselection_triggered": triggered,
            "switch_accepted": switch,
            "trigger_reasons": ";".join(reasons),
            "ranked_families": ";".join(ranked),
            "validated_families": ";".join(validated),
            "log_marginal_likelihood": selected.log_marginal_likelihood,
            "validation_score": selected_score,
            "challenger_validation_score": challenger_score,
            "score_improvement": improvement,
            "length_scale_x": scales[0],
            "length_scale_y": scales[1],
            "optimization_runtime": optimization_runtime,
            "family_fit_count": len(KERNEL_FAMILIES) if triggered else 1,
        }
        return SelectionOutcome(
            selected,
            selected_family,
            float(selected_score),
            triggered,
            switch,
            retained,
            tuple(reasons),
            previous,
            float(optimization_runtime),
            event_row,
        )

    def _fit_family(
        self,
        family: str,
        X: np.ndarray,
        y: np.ndarray,
        warm_kernel: Any | None,
    ) -> GPRSurrogate:
        self.optimization_count += 1
        kernel = clone(warm_kernel) if warm_kernel is not None else build_named_kernel(family, 2)
        config = _fit_config(self.base_config, kernel, self.random_state)
        return GPRSurrogate(config).fit(X, y)


def run_monte_carlo_2d(
    config_path: Path,
    output_root: Path,
    *,
    smoke: bool = False,
) -> Path:
    """Run the configured study, or its smoke trial count when requested."""

    config = load_config(config_path)
    _validate_monte_carlo_config(config)
    trials = int(config["benchmark"].get("smoke_trials", 5) if smoke else config["trials"])
    config["trials"] = trials
    config["benchmark"]["trials"] = trials
    output = prepare_suite_output(output_root, OUTPUT_NAME)
    write_resolved_config(output, config)
    for directory in (output / "metrics", output / "kernel", output / "figures", output / "gifs"):
        directory.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    per_step: list[dict[str, Any]] = []
    per_trial: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    trajectories: dict[str, list[TrajectoryState]] = {}
    field_index = {name: index for index, name in enumerate(FIELD_NAMES)}
    for field_name in FIELD_NAMES:
        for trial in range(trials):
            scenario = _make_scenario(config, field_name, trial, field_index[field_name])
            for method in METHOD_NAMES:
                trial_rows, trial_events, trajectory = _run_method(
                    scenario,
                    method,
                    config,
                    capture_trajectory=method in ADAPTIVE_METHOD_NAMES and trial == 0,
                )
                per_step.extend(trial_rows)
                events.extend(
                    {
                        "field": field_name,
                        "trial": trial,
                        "method": method,
                        **row,
                    }
                    for row in trial_events
                )
                per_trial.append(_trial_record(field_name, trial, method, trial_rows, trial_events))
                if trajectory is not None:
                    trajectories[f"{field_name}::{method}"] = trajectory

    paired = _paired_records(per_trial)
    _write_csv(output / "metrics" / "per_step.csv", per_step, PER_STEP_FIELDS)
    _write_csv(output / "metrics" / "per_trial.csv", per_trial, TRIAL_FIELDS)
    _write_csv(output / "metrics" / "paired_advantage.csv", paired, PAIRED_FIELDS)
    _write_csv(output / "kernel" / "events.csv", events, EVENT_FIELDS)
    _write_learning_curve_figure(output / "figures" / "r2_learning_curves.png", per_step, per_trial)
    _write_paired_figure(output / "figures" / "paired_adaptive_advantage.png", paired)
    for trajectory_key, trajectory in trajectories.items():
        field_name, method = trajectory_key.split("::", 1)
        filename = (
            f"{field_name}.gif"
            if method == "adaptive_krispu"
            else f"{field_name}_{method}.gif"
        )
        _write_gif(output / "gifs" / filename, trajectory, int(config["dpi"]))
    runtime = perf_counter() - started
    _write_report(output / "report.md", per_trial, events, runtime)
    return output


def _validate_monte_carlo_config(config: dict[str, Any]) -> None:
    if tuple(config["fields"]) != FIELD_NAMES:
        raise ValueError(f"The 2D study must use exactly these fields: {FIELD_NAMES}")
    if tuple(config["methods"]) != METHOD_NAMES:
        raise ValueError(f"The 2D study must use exactly these methods: {METHOD_NAMES}")
    if int(config["initial_sample_count"]) < 4:
        raise ValueError("initial_sample_count must allow an eligible buffered jackknife.")
    if int(config["final_budget"]) <= int(config["initial_sample_count"]):
        raise ValueError("final_budget must exceed initial_sample_count.")
    if int(config["gif_grid_size"]) < 2:
        raise ValueError("gif_grid_size must be at least 2.")


def _make_scenario(
    config: dict[str, Any], field_name: str, trial: int, field_index: int
) -> Scenario:
    base_seed = int(config["base_seed"])
    seed = base_seed + field_index * 100_000 + trial * 1_000
    rng = np.random.default_rng(seed)
    transformation = (
        {
            "rotation_degrees": 0.0,
            "axis_scale": (1.0, 1.0),
            "translation": (0.0, 0.0),
            "output_scale": 1.0,
        }
        if field_name == "matched_spherical"
        else {
            "rotation_degrees": float(rng.uniform(-35.0, 35.0)),
            "axis_scale": tuple(float(value) for value in rng.uniform(0.75, 1.30, 2)),
            "translation": tuple(float(value) for value in rng.uniform(-0.08, 0.08, 2)),
            "output_scale": float(rng.uniform(0.75, 1.35)),
        }
    )
    field = _make_trial_field(field_name, transformation, seed + 17)
    domain = field.domain
    initial_seed = seed + 1
    candidate_seed = seed + 2
    initial_name = "clustered_observations" if field_name == "clustered_observations" else "interior_maximin"
    initial_X = initial_design(
        initial_name,
        domain,
        int(config["initial_sample_count"]),
        float(config["initial_boundary_margin"]),
        initial_seed,
    )
    candidate_pool = generate_candidates(
        domain, int(config["candidate_count"]), "lhs", candidate_seed
    )
    evaluation_points = _regular_grid(domain, int(config["evaluation_grid_size"]))
    visualization_points = _regular_grid(domain, int(config["gif_grid_size"]))
    return Scenario(
        field,
        trial,
        seed,
        initial_X,
        candidate_pool,
        evaluation_points,
        field.clean(evaluation_points),
        visualization_points,
        field.clean(visualization_points),
    )


def _make_trial_field(
    name: str, transformation: dict[str, Any], noise_seed: int
) -> TrialField:
    domain = ContinuousDomain([[-1.0, 1.0], [-1.0, 1.0]], names=("x", "y"))
    if name == "matched_spherical":
        return _make_matched_spherical_field(noise_seed, domain)
    base_function = _base_function(name)
    angle = np.deg2rad(float(transformation["rotation_degrees"]))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    axis_scale = np.asarray(transformation["axis_scale"], dtype=float)
    translation = np.asarray(transformation["translation"], dtype=float)
    output_scale = float(transformation["output_scale"])
    noise_cache: dict[tuple[int, int], float] = {}

    def source_points(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float).reshape(-1, 2)
        normalized = (values + 1.0) / 2.0
        centered = (normalized - 0.5 - translation) / axis_scale
        return np.clip(centered @ rotation.T + 0.5, 0.0, 1.0) * 2.0 - 1.0

    def clean(points: np.ndarray) -> np.ndarray:
        return output_scale * np.asarray(base_function(source_points(points)), dtype=float)

    def noise(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float).reshape(-1, 2)
        result = np.empty(len(values), dtype=float)
        for index, point in enumerate(values):
            key = tuple(np.rint((point + 1.0) * 500_000.0).astype(int))
            if key not in noise_cache:
                rng = np.random.default_rng(
                    np.random.SeedSequence([noise_seed & 0xFFFFFFFF, key[0], key[1]])
                )
                noise_cache[key] = float(rng.normal())
            result[index] = noise_cache[key]
        if name == "white_noise":
            scale = np.full(len(values), 0.30 * output_scale)
        else:
            scale = output_scale * (0.10 + 0.35 * ((values[:, 0] + 1.0) / 2.0))
        return result * scale

    observed = clean if name not in {"white_noise", "heteroscedastic_noise"} else lambda points: clean(points) + noise(points)
    metadata = {
        "transformation": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in transformation.items()
        },
        "noise_seed": noise_seed,
        "noisy_observations": name in {"white_noise", "heteroscedastic_noise"},
    }
    return TrialField(name, domain, clean, observed, metadata)


def _make_matched_spherical_field(seed: int, domain: ContinuousDomain) -> TrialField:
    """Return a locked spherical-kernel draw matching the kriging family."""

    sampled = sampled_gp_field(
        family="spherical_ard",
        amplitude=1.0,
        length_scales=(0.60, 0.60),
        nugget=1.0e-6,
        seed=seed,
    )

    def clean(points: np.ndarray) -> np.ndarray:
        return np.asarray(sampled.evaluate(points), dtype=float)

    return TrialField(
        "matched_spherical",
        domain,
        clean,
        clean,
        {
            "true_kernel": {
                "family": "spherical_ard",
                "amplitude": 1.0,
                "ard_length_scales": [0.60, 0.60],
                "nugget": 1.0e-6,
            },
            "kriging_kernel_family": "spherical_ard",
            "fixed_kriging_kernel": "spherical_ard",
            "kernel_match": "kriging starts from spherical_ard",
            "seed": seed,
        },
    )


def _base_function(name: str) -> Callable[[np.ndarray], np.ndarray]:
    def smooth(points: np.ndarray) -> np.ndarray:
        x, y = points[:, 0], points[:, 1]
        return (
            0.60 * np.sin(2.5 * x)
            - 0.40 * np.cos(2.0 * y)
            - 0.25 * x * y
            - 0.50 * np.exp(-3.0 * ((x - 0.35) ** 2 + (y + 0.25) ** 2))
        )

    if name == "smooth_single_scale":
        return lambda points: 0.65 * np.sin(1.8 * points[:, 0]) + 0.35 * np.cos(1.4 * points[:, 1])
    if name == "clustered_observations":
        return lambda points: (
            0.20 * points[:, 0]
            + 0.15 * points[:, 1]
            - 0.30 * np.sin(2.0 * points[:, 0])
            - 1.40
            * np.exp(-((points[:, 0] - 0.35) ** 2 / 0.025) - ((points[:, 1] + 0.2) ** 2 / 0.04))
        )
    if name == "branin":
        return _scaled_branin
    if name == "baseline_drift":
        return lambda points: smooth(points) + 0.70 * points[:, 0]
    if name in {"white_noise", "heteroscedastic_noise"}:
        return _deterministic_trend
    raise ValueError(f"Unknown 2D field {name!r}.")


def _deterministic_trend(points: np.ndarray) -> np.ndarray:
    """Low-frequency latent trend used by both noisy benchmark fields."""

    x, y = points[:, 0], points[:, 1]
    return 0.55 * np.sin(1.6 * x) + 0.35 * np.cos(1.2 * y) + 0.20 * x - 0.15 * y


def _scaled_branin(points: np.ndarray) -> np.ndarray:
    x = 7.5 * points[:, 0] + 2.5
    y = 7.5 * points[:, 1] + 7.5
    raw = (y - 5.1 / (4.0 * np.pi**2) * x**2 + 5.0 / np.pi * x - 6.0) ** 2
    raw += 10.0 * (1.0 - 1.0 / (8.0 * np.pi)) * np.cos(x) + 10.0
    return (raw - 50.0) / 45.0


def _regular_grid(domain: ContinuousDomain, size: int) -> np.ndarray:
    axes = [np.linspace(lo, hi, size) for lo, hi in domain.bounds]
    mesh = np.meshgrid(*axes, indexing="xy")
    return np.column_stack([values.ravel() for values in mesh])


def _run_method(
    scenario: Scenario,
    method: str,
    config: dict[str, Any],
    *,
    capture_trajectory: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[TrajectoryState] | None]:
    domain = scenario.field.domain
    observed_X = scenario.initial_X.copy()
    observed_y = scenario.field.observed(observed_X)
    available = np.ones(len(scenario.candidate_pool), dtype=bool)
    normalized_candidates = domain.normalize(scenario.candidate_pool)
    normalized_evaluation = domain.normalize(scenario.evaluation_points)
    normalized_visualization = domain.normalize(scenario.visualization_points)
    method_seed = scenario.seed + 101
    base_config = _gpr_config(config, noisy=scenario.field.name in {"white_noise", "heteroscedastic_noise"})
    controller = (
        AdaptiveKernelController(
            method_seed,
            base_config,
            initial_family=str(
                scenario.field.metadata.get("kriging_kernel_family", "matern_32_ard")
            ),
            fixed_family=scenario.field.metadata.get("fixed_kriging_kernel"),
        )
        if method in ADAPTIVE_METHOD_NAMES
        else None
    )
    baseline_order = _method_order(method, len(available), method_seed)
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    trajectory: list[TrajectoryState] | None = [] if capture_trajectory else None
    newest = observed_X[-1].copy()
    final_budget = int(config["final_budget"])
    for sample_count in range(len(observed_X), final_budget + 1):
        started = perf_counter()
        X_normalized = domain.normalize(observed_X)
        if controller is not None:
            outcome = controller.select(X_normalized, observed_y, sample_count)
            surrogate = outcome.surrogate
            event = dict(outcome.event_row)
            event["sample_count"] = sample_count
            event_rows.append(event)
            reference_points = normalized_visualization if capture_trajectory else normalized_evaluation
            combined = np.vstack((reference_points, normalized_candidates))
            plan = _jackknife_plan(X_normalized, base_config)
            if method == "adaptive_krispu":
                evaluation_acquisition, candidate_acquisition = _adaptive_acquisition(
                    surrogate, X_normalized, observed_y, combined, plan, len(reference_points)
                )
            elif method == "adaptive_kriging":
                evaluation_acquisition, candidate_acquisition = (
                    _jackknife_uncertainty_acquisition(
                        surrogate,
                        X_normalized,
                        observed_y,
                        combined,
                        plan,
                        len(reference_points),
                    )
                )
            else:
                evaluation_acquisition, candidate_acquisition = (
                    _raw_jackknife_uncertainty_acquisition(
                        surrogate,
                        X_normalized,
                        observed_y,
                        combined,
                        len(reference_points),
                    )
                )
            scores = candidate_acquisition
            selection_kernel = outcome.family
            reselection = outcome.reselection_triggered
            switch = outcome.switch_accepted
        else:
            surrogate = _fit_baseline(
                X_normalized,
                observed_y,
                base_config,
                method_seed,
            )
            evaluation_acquisition = np.zeros(len(normalized_evaluation), dtype=float)
            scores = np.zeros(len(normalized_candidates), dtype=float)
            selection_kernel = "matern_32_ard"
            reselection = False
            switch = False

        predicted, _ = surrogate.predict(normalized_evaluation)
        metrics = reconstruction_metrics(scenario.evaluation_truth, predicted)
        next_index: int | None = None
        if sample_count < final_budget:
            next_index = _select_candidate(
                method,
                scores,
                normalized_candidates,
                X_normalized,
                observed_X,
                available,
                domain,
                baseline_order,
                minimum_normalized_distance(config),
            )
            available[next_index] = False
            newest = scenario.candidate_pool[next_index].copy()
        runtime = perf_counter() - started
        scales = _length_scales(surrogate.frozen_kernel)
        rows.append(
            {
                "field": scenario.field.name,
                "trial": scenario.trial,
                "method": method,
                "sample_count": sample_count,
                "r2": metrics.r2,
                "nrmse": metrics.nrmse,
                "runtime": runtime,
                "selected_kernel": selection_kernel,
                "length_scale_x": scales[0],
                "length_scale_y": scales[1],
                "reselection_triggered": reselection,
                "switch_accepted": switch,
            }
        )
        if capture_trajectory:
            assert trajectory is not None
            map_prediction, _ = surrogate.predict(normalized_visualization)
            trajectory.append(TrajectoryState(
                scenario.visualization_truth.copy(),
                map_prediction.copy(),
                evaluation_acquisition.copy(),
                scenario.visualization_points.copy(),
                observed_X.copy(),
                newest.copy(),
                scenario.field.name,
                sample_count,
                metrics.r2,
                method,
            ))
        if next_index is not None:
            observed_X = np.vstack((observed_X, scenario.candidate_pool[next_index]))
            observed_y = np.concatenate(
                (observed_y, scenario.field.observed(scenario.candidate_pool[next_index : next_index + 1]))
            )
    return rows, event_rows, trajectory


def _gpr_config(config: dict[str, Any], *, noisy: bool) -> GPRConfig:
    del noisy
    return GPRConfig(
        alpha=1.0e-6,
        n_restarts_optimizer=0,
        random_state=int(config["base_seed"]),
        jackknife=jackknife_config(config),
        optimize_hyperparameters=True,
    )


def _fit_config(base: GPRConfig, kernel: Any, random_state: int) -> GPRConfig:
    return GPRConfig(
        **{
            **base.__dict__,
            "kernel": kernel,
            "optimize_hyperparameters": True,
            "n_restarts_optimizer": 0,
            "random_state": random_state,
        }
    )


def _fit_baseline(
    X: np.ndarray, y: np.ndarray, base_config: GPRConfig, random_state: int
) -> GPRSurrogate:
    kernel = build_named_kernel("matern_32_ard", 2)
    return GPRSurrogate(_fit_config(base_config, kernel, random_state)).fit(X, y)


def _jackknife_plan(X: np.ndarray, config: GPRConfig) -> BufferedJackknifePlan | None:
    minimum = config.jackknife.minimum_training_points
    if len(X) < minimum + 1:
        return None
    return build_buffered_jackknife_plan(
        X,
        multiplier=config.jackknife.multiplier,
        minimum_radius=config.jackknife.minimum_radius,
        maximum_radius=config.jackknife.maximum_radius,
        minimum_training_points=minimum,
    )


def _buffered_validation_score(
    surrogate: GPRSurrogate,
    X: np.ndarray,
    y: np.ndarray,
    plan: BufferedJackknifePlan,
) -> float:
    scores: list[float] = []
    for anchor, removed in zip(plan.anchor_indices, plan.removed_indices_by_fold, strict=True):
        keep = np.ones(len(X), dtype=bool)
        keep[removed] = False
        fold = GPRSurrogate(surrogate.config).fit_fixed_kernel(
            X[keep], y[keep], frozen_kernel=surrogate.frozen_kernel
        )
        predicted, standard = fold.predict(X[anchor : anchor + 1])
        deviation = max(float(standard[0]), surrogate.config.response_epsilon)
        residual = float(y[anchor] - predicted[0])
        scores.append(0.5 * np.log(2.0 * np.pi * deviation**2) + residual**2 / (2.0 * deviation**2))
    return float(np.mean(scores)) if scores else np.inf


def _adaptive_acquisition(
    surrogate: GPRSurrogate,
    X: np.ndarray,
    y: np.ndarray,
    references: np.ndarray,
    plan: BufferedJackknifePlan | None,
    evaluation_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    evaluation, candidates = _jackknife_uncertainty_acquisition(
        surrogate, X, y, references, plan, evaluation_count
    )
    support, _ = kernel_support_deficit(surrogate, X, references)
    acquisition = np.concatenate((evaluation, candidates)) * np.sqrt(
        np.maximum(support, 0.0)
    )
    return acquisition[:evaluation_count], acquisition[evaluation_count:]


def _jackknife_uncertainty_acquisition(
    surrogate: GPRSurrogate,
    X: np.ndarray,
    y: np.ndarray,
    references: np.ndarray,
    plan: BufferedJackknifePlan | None,
    evaluation_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use buffered-jackknife field sensitivity without support weighting."""

    if plan is None:
        return np.zeros(evaluation_count), np.zeros(len(references) - evaluation_count)
    field_means: list[np.ndarray] = []
    for removed in plan.removed_indices_by_fold:
        keep = np.ones(len(X), dtype=bool)
        keep[removed] = False
        fold = GPRSurrogate(surrogate.config).fit_fixed_kernel(
            X[keep], y[keep], frozen_kernel=surrogate.frozen_kernel
        )
        mean, _ = fold.predict(references)
        field_means.append(mean)
    _, sensitivity = jackknife_field_sensitivity(np.column_stack(field_means))
    return sensitivity[:evaluation_count], sensitivity[evaluation_count:]


def _raw_jackknife_uncertainty_acquisition(
    surrogate: GPRSurrogate,
    X: np.ndarray,
    y: np.ndarray,
    references: np.ndarray,
    evaluation_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use ordinary leave-one-out jackknife sensitivity only."""

    if len(X) < 2:
        return np.zeros(evaluation_count), np.zeros(len(references) - evaluation_count)
    field_means: list[np.ndarray] = []
    for removed_index in range(len(X)):
        keep = np.ones(len(X), dtype=bool)
        keep[removed_index] = False
        fold = GPRSurrogate(surrogate.config).fit_fixed_kernel(
            X[keep], y[keep], frozen_kernel=surrogate.frozen_kernel
        )
        mean, _ = fold.predict(references)
        field_means.append(mean)
    _, sensitivity = jackknife_field_sensitivity(np.column_stack(field_means))
    return sensitivity[:evaluation_count], sensitivity[evaluation_count:]


def _method_order(method: str, count: int, seed: int) -> np.ndarray:
    if method == "random":
        return np.random.default_rng(seed).permutation(count)
    return np.arange(count, dtype=int)


def _select_candidate(
    method: str,
    scores: np.ndarray,
    candidates_normalized: np.ndarray,
    observed_normalized: np.ndarray,
    observed_physical: np.ndarray,
    available: np.ndarray,
    domain: ContinuousDomain,
    baseline_order: np.ndarray,
    minimum_distance: float,
) -> int:
    valid = valid_candidate_mask(
        domain,
        domain.denormalize(candidates_normalized),
        observed_physical,
        minimum_normalized_distance=minimum_distance,
    )
    valid &= available
    if not np.any(valid):
        raise RuntimeError("The shared candidate pool was exhausted before the point budget.")
    if method in {
        "adaptive_krispu",
        "adaptive_kriging",
        "raw_jackknife_sensitivity",
        "progressive_lhs",
    }:
        masked = np.where(valid, scores, -np.inf)
        return int(np.argmax(masked))
    if method == "random":
        for index in baseline_order:
            if valid[index]:
                return int(index)
    distances = np.linalg.norm(
        candidates_normalized[:, None, :] - observed_normalized[None, :, :], axis=2
    ).min(axis=1)
    return int(np.argmax(np.where(valid, distances, -np.inf)))


def _length_scales(kernel: Any) -> tuple[float, float]:
    values = np.asarray(kernel.get_params(deep=False).get("length_scale"), dtype=float).reshape(-1)
    if values.shape != (2,):
        for name, parameter in kernel.get_params(deep=True).items():
            if name.endswith("length_scale"):
                values = np.asarray(parameter, dtype=float).reshape(-1)
                break
    if values.shape != (2,):
        return (float("nan"), float("nan"))
    return float(values[0]), float(values[1])


def _length_scale_near_bound(kernel: Any) -> bool:
    for parameter in kernel.hyperparameters:
        if parameter.name.endswith("length_scale"):
            values = np.asarray(kernel.get_params(deep=True)[parameter.name], dtype=float).reshape(-1)
            bounds = np.asarray(parameter.bounds, dtype=float)
            return bool(np.any(values <= bounds[:, 0] * 1.05) or np.any(values >= bounds[:, 1] * 0.95))
    return False


def _trial_record(
    field: str, trial: int, method: str, rows: list[dict[str, Any]], events: list[dict[str, Any]]
) -> dict[str, Any]:
    counts = np.asarray([row["sample_count"] for row in rows], dtype=float)
    r2 = np.asarray([row["r2"] for row in rows], dtype=float)
    nrmse = np.asarray([row["nrmse"] for row in rows], dtype=float)
    checks = sum(bool(event["family_check"]) for event in events)
    switches = sum(bool(event["switch_accepted"]) for event in events)
    return {
        "field": field,
        "trial": trial,
        "method": method,
        "final_r2": float(r2[-1]),
        "r2_auc": float(np.trapz(r2, counts)),
        "final_nrmse": float(nrmse[-1]),
        "runtime": float(sum(float(row["runtime"]) for row in rows)),
        "reselection_count": checks,
        "switch_count": switches,
    }


def _paired_records(per_trial: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(row["field"], row["trial"], row["method"]): row for row in per_trial}
    rows: list[dict[str, Any]] = []
    field_trials = sorted({(row["field"], row["trial"]) for row in per_trial})
    for field, trial in field_trials:
        for adaptive_method in ADAPTIVE_METHOD_NAMES:
            adaptive = index[(field, trial, adaptive_method)]
            for baseline in BASELINE_METHOD_NAMES:
                control = index[(field, trial, baseline)]
                for metric in ("r2_auc", "final_r2"):
                    adaptive_value = float(adaptive[metric])
                    baseline_value = float(control[metric])
                    rows.append(
                        {
                            "field": field,
                            "trial": trial,
                            "adaptive_method": adaptive_method,
                            "baseline": baseline,
                            "metric": metric,
                            "adaptive_value": adaptive_value,
                            "baseline_value": baseline_value,
                            "adaptive_minus_baseline": adaptive_value - baseline_value,
                        }
                    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_learning_curve_figure_old(
    path: Path, rows: list[dict[str, Any]], trials: list[dict[str, Any]]
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    colors = dict(
        zip(
            METHOD_NAMES,
            ("#b2182b", "#e66101", "#762a83", "#2166ac", "#1b7837", "#4d4d4d"),
            strict=True,
        )
    )
    for axis, field in zip(axes.flat, FIELD_NAMES, strict=True):
        for method in METHOD_NAMES:
            values = [row for row in rows if row["field"] == field and row["method"] == method]
            sample_counts = sorted({int(row["sample_count"]) for row in values})
            medians = []
            lower = []
            upper = []
            for count in sample_counts:
                r2 = [float(row["r2"]) for row in values if int(row["sample_count"]) == count]
                medians.append(float(np.median(r2)))
                lower.append(float(np.percentile(r2, 10)))
                upper.append(float(np.percentile(r2, 90)))
            axis.plot(sample_counts, medians, color=colors[method], label=method, linewidth=1.7)
            axis.fill_between(sample_counts, lower, upper, color=colors[method], alpha=0.10)
        adaptive = [
            row
            for row in trials
            if row["field"] == field and row["method"] in ADAPTIVE_METHOD_NAMES
        ]
        frequency = float(np.mean([int(row["switch_count"]) > 0 for row in adaptive])) if adaptive else 0.0
        axis.set_title(field.replace("_", " "))
        axis.set_xlabel("sample count")
        axis.set_ylabel("R²")
        axis.grid(alpha=0.25)
        axis.text(0.03, 0.96, f"adaptive switch frequency: {frequency:.1%}", transform=axis.transAxes, fontsize=8, va="top")
    axes.flat[0].legend(fontsize=8, loc="best")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_learning_curve_figure(
    path: Path, rows: list[dict[str, Any]], trials: list[dict[str, Any]]
) -> None:
    columns = 3
    row_count = int(np.ceil(len(FIELD_NAMES) / columns))
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(14, 4.0 * row_count),
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)
    colors = dict(
        zip(
            METHOD_NAMES,
            ("#b2182b", "#e66101", "#762a83", "#2166ac", "#1b7837", "#4d4d4d"),
            strict=True,
        )
    )
    for axis, field in zip(axes, FIELD_NAMES, strict=False):
        for method in METHOD_NAMES:
            values = [
                row for row in rows if row["field"] == field and row["method"] == method
            ]
            sample_counts = sorted({int(row["sample_count"]) for row in values})
            medians = []
            lower = []
            upper = []
            for count in sample_counts:
                r2 = [float(row["r2"]) for row in values if int(row["sample_count"]) == count]
                medians.append(float(np.median(r2)))
                lower.append(float(np.percentile(r2, 10)))
                upper.append(float(np.percentile(r2, 90)))
            axis.plot(sample_counts, medians, color=colors[method], label=method, linewidth=1.7)
            axis.fill_between(sample_counts, lower, upper, color=colors[method], alpha=0.10)
        adaptive = [
            row
            for row in trials
            if row["field"] == field and row["method"] in ADAPTIVE_METHOD_NAMES
        ]
        frequency = (
            float(np.mean([int(row["switch_count"]) > 0 for row in adaptive]))
            if adaptive
            else 0.0
        )
        axis.set_title(field.replace("_", " "))
        axis.set_xlabel("sample count")
        axis.set_ylabel("R^2")
        axis.grid(alpha=0.25)
        axis.text(
            0.03,
            0.96,
            f"adaptive switch frequency: {frequency:.1%}",
            transform=axis.transAxes,
            fontsize=8,
            va="top",
        )
    for axis in axes[len(FIELD_NAMES) :]:
        axis.axis("off")
    axes[0].legend(fontsize=8, loc="best")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_paired_figure_old(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for axis, metric in zip(axes, ("r2_auc", "final_r2"), strict=True):
        positions = np.arange(len(FIELD_NAMES) * 3)
        labels: list[str] = []
        values: list[np.ndarray] = []
        for field in FIELD_NAMES:
            for baseline in ("progressive_lhs", "sequential_maximin", "random"):
                subset = np.asarray(
                    [row["adaptive_minus_baseline"] for row in rows if row["field"] == field and row["baseline"] == baseline and row["metric"] == metric],
                    dtype=float,
                )
                values.append(subset)
                labels.append(f"{field.replace('_', ' ')}\n{baseline.replace('_', ' ')}")
        medians = [float(np.median(value)) for value in values]
        lower = [float(np.percentile(value, 5)) for value in values]
        upper = [float(np.percentile(value, 95)) for value in values]
        axis.errorbar(positions, medians, yerr=[np.asarray(medians) - lower, np.asarray(upper) - medians], fmt="o", color="#b2182b", capsize=3)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(positions, labels, rotation=70, ha="right", fontsize=7)
        axis.set_ylabel("adaptive KRISP-U minus baseline")
        axis.set_title("R² AUC" if metric == "r2_auc" else "final R²")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_paired_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    groups = [(field, baseline) for field in FIELD_NAMES for baseline in BASELINE_METHOD_NAMES]
    positions = np.arange(len(groups), dtype=float)
    labels = [
        f"{field.replace('_', ' ')}\n{baseline.replace('_', ' ')}"
        for field, baseline in groups
    ]
    colors = {
        "adaptive_krispu": "#b2182b",
        "adaptive_kriging": "#e66101",
        "raw_jackknife_sensitivity": "#762a83",
    }
    for axis, metric in zip(axes, ("r2_auc", "final_r2"), strict=True):
        offsets = np.linspace(-0.20, 0.20, len(ADAPTIVE_METHOD_NAMES))
        for offset, adaptive_method in zip(offsets, ADAPTIVE_METHOD_NAMES, strict=True):
            values = []
            for field, baseline in groups:
                subset = np.asarray(
                    [
                        row["adaptive_minus_baseline"]
                        for row in rows
                        if row["field"] == field
                        and row["adaptive_method"] == adaptive_method
                        and row["baseline"] == baseline
                        and row["metric"] == metric
                    ],
                    dtype=float,
                )
                values.append(subset)
            medians = [float(np.median(value)) for value in values]
            lower = [float(np.percentile(value, 5)) for value in values]
            upper = [float(np.percentile(value, 95)) for value in values]
            axis.errorbar(
                positions + offset,
                medians,
                yerr=[np.asarray(medians) - lower, np.asarray(upper) - medians],
                fmt="o",
                color=colors[adaptive_method],
                capsize=3,
                label=adaptive_method,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(positions, labels, rotation=70, ha="right", fontsize=7)
        axis.set_ylabel("adaptive method minus baseline")
        axis.set_title("RÂ² AUC" if metric == "r2_auc" else "final RÂ²")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_gif(path: Path, trajectory: list[TrajectoryState], dpi: int) -> None:
    temporary = Path(tempfile.mkdtemp(prefix="krispu_frames_", dir=path.parent))
    frames: list[Image.Image] = []
    for frame_number, current in enumerate(trajectory):
        size = round(np.sqrt(len(current.truth)))
        truth = current.truth.reshape(size, size)
        prediction = current.prediction.reshape(size, size)
        uncertainty = current.uncertainty.reshape(size, size)
        error = np.abs(truth - prediction)
        limits = (float(np.min(truth)), float(np.max(truth)))
        figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
        for axis, values, title in zip(
            axes.flat,
            (truth, prediction, uncertainty, error),
            (
                "truth",
                "reconstruction",
                (
                    "KRISP-U uncertainty"
                    if current.method == "adaptive_krispu"
                    else "GP posterior standard deviation"
                ),
                "absolute error",
            ),
            strict=True,
        ):
            if title in {"truth", "reconstruction"}:
                image = axis.imshow(values, origin="lower", extent=(-1, 1, -1, 1), vmin=limits[0], vmax=limits[1], cmap="viridis")
            else:
                image = axis.imshow(values, origin="lower", extent=(-1, 1, -1, 1), cmap="magma")
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.scatter(current.observed_X[:, 0], current.observed_X[:, 1], s=13, c="white", edgecolors="black", linewidths=0.4)
            axis.scatter([current.newest_point[0]], [current.newest_point[1]], s=45, c="#e41a1c", edgecolors="black", linewidths=0.6)
            del image
        annotation = (
            f"{current.field} | {current.method} | points={current.sample_count} "
            f"| R²={current.r2:.3f}"
        )
        figure.suptitle(annotation, fontsize=10)
        frame_path = temporary / f"frame_{frame_number:03d}.png"
        figure.savefig(frame_path, dpi=dpi)
        plt.close(figure)
        frames.append(Image.open(frame_path).convert("P", palette=Image.Palette.ADAPTIVE))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=450, loop=0, optimize=False)
    for frame in frames:
        frame.close()
    shutil.rmtree(temporary)


def _write_report_old(path: Path, trials: list[dict[str, Any]], events: list[dict[str, Any]], runtime: float) -> None:
    adaptive = [row for row in trials if row["method"] == "adaptive_krispu"]
    total_checks = sum(bool(row["family_check"]) for row in events)
    total_switches = sum(bool(row["switch_accepted"]) for row in events)
    adaptive_steps = len(events)
    adaptive_optimizations = sum(int(row["family_fit_count"]) for row in events)
    total_optimizations = adaptive_optimizations + 3 * adaptive_steps
    lines = [
        "# KRISP-U v0.3.0 2D Monte Carlo study",
        "",
        f"Total runtime: {runtime:.2f} s; runtime per adaptive step: {sum(row['runtime'] for row in adaptive) / max(adaptive_steps, 1):.4f} s.",
        f"Kernel optimizations: {total_optimizations} total ({adaptive_optimizations} adaptive); family checks: {total_checks}; accepted switches: {total_switches}.",
        "",
        "## Median final R² and R² AUC",
        "",
        "| Field | Method | final R² | R² AUC |",
        "|---|---|---:|---:|",
    ]
    for field in FIELD_NAMES:
        for method in METHOD_NAMES:
            subset = [row for row in trials if row["field"] == field and row["method"] == method]
            lines.append(f"| {field} | {method} | {np.median([row['final_r2'] for row in subset]):.5g} | {np.median([row['r2_auc'] for row in subset]):.5g} |")
    lines.extend(("", "## Paired adaptive intervals", "", "| Field | Baseline | Metric | median Δ | 5th–95th percentile |", "|---|---|---|---:|---:|"))
    paired = _paired_records(trials)
    for field in FIELD_NAMES:
        for baseline in ("progressive_lhs", "sequential_maximin", "random"):
            for metric in ("r2_auc", "final_r2"):
                values = np.asarray([row["adaptive_minus_baseline"] for row in paired if row["field"] == field and row["baseline"] == baseline and row["metric"] == metric], dtype=float)
                lines.append(f"| {field} | {baseline} | {metric} | {np.median(values):.5g} | [{np.percentile(values, 5):.5g}, {np.percentile(values, 95):.5g}] |")
    lines.extend(("", "The noisy fields were fitted from noisy observations and scored against their clean latent fields. R² AUC is trapezoidal integration over sample count."))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    trials: list[dict[str, Any]],
    events: list[dict[str, Any]],
    runtime: float,
) -> None:
    adaptive = [row for row in trials if row["method"] in ADAPTIVE_METHOD_NAMES]
    total_checks = sum(bool(row["family_check"]) for row in events)
    total_switches = sum(bool(row["switch_accepted"]) for row in events)
    adaptive_steps = len(events)
    adaptive_optimizations = sum(int(row["family_fit_count"]) for row in events)
    total_optimizations = adaptive_optimizations + 3 * adaptive_steps
    lines = [
        "# KRISP-U v0.3.0 2D Monte Carlo study",
        "",
        (
            f"Total runtime: {runtime:.2f} s; runtime per adaptive step: "
            f"{sum(row['runtime'] for row in adaptive) / max(adaptive_steps, 1):.4f} s."
        ),
        (
            f"Kernel optimizations: {total_optimizations} total "
            f"({adaptive_optimizations} adaptive); family checks: {total_checks}; "
            f"accepted switches: {total_switches}."
        ),
        "",
        "## Median final R^2 and R^2 AUC",
        "",
        "| Field | Method | final R^2 | R^2 AUC |",
        "|---|---|---:|---:|",
    ]
    for field in FIELD_NAMES:
        for method in METHOD_NAMES:
            subset = [row for row in trials if row["field"] == field and row["method"] == method]
            lines.append(
                f"| {field} | {method} | {np.median([row['final_r2'] for row in subset]):.5g} | "
                f"{np.median([row['r2_auc'] for row in subset]):.5g} |"
            )
    lines.extend(
        (
            "",
            "## Paired adaptive intervals",
            "",
            "| Field | Adaptive method | Baseline | Metric | median delta | 5th-95th percentile |",
            "|---|---|---|---:|---:|---:|",
        )
    )
    paired = _paired_records(trials)
    for field in FIELD_NAMES:
        for adaptive_method in ADAPTIVE_METHOD_NAMES:
            for baseline in BASELINE_METHOD_NAMES:
                for metric in ("r2_auc", "final_r2"):
                    values = np.asarray(
                        [
                            row["adaptive_minus_baseline"]
                            for row in paired
                            if row["field"] == field
                            and row["adaptive_method"] == adaptive_method
                            and row["baseline"] == baseline
                            and row["metric"] == metric
                        ],
                        dtype=float,
                    )
                    lines.append(
                        f"| {field} | {adaptive_method} | {baseline} | {metric} | "
                        f"{np.median(values):.5g} | "
                        f"[{np.percentile(values, 5):.5g}, "
                        f"{np.percentile(values, 95):.5g}] |"
                    )
    lines.extend(
        (
            "",
            (
                "Adaptive KRISP-U uses jackknife sensitivity multiplied by a kernel-support "
                "weight. Adaptive kriging uses buffered-jackknife sensitivity alone; it "
                "does not apply the support-deficit multiplier or posterior standard deviation. "
                "The raw-jackknife variant uses ordinary leave-one-out sensitivity without "
                "buffering, support weighting, or posterior standard deviation."
            ),
            (
                "The matched_spherical field is a smoother, long-correlation spherical_ard "
                "draw; adaptive kriging controllers are fixed to spherical_ard for this case."
            ),
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["run_monte_carlo_2d"]
