# KRISP-U

KRISP-U is a Gaussian Process Regression (GPR) package for active field
reconstruction. Its main workflow is: give KRISP-U the measurements you already
have, let it model the response field, and ask which point or preset candidate
should be measured next to reduce uncertainty in the field.

It supports continuous bounded domains, preset candidate pools, and mixed
continuous/discrete candidate spaces. The default acquisition is maximum
predictive uncertainty, which matches the field-learning use case. Improvement
acquisitions remain available for toy optimization comparisons. For the original
KRISP-style field-information objective, use `acquisition="kld"`; this scores a
candidate by the expected KL information gain it provides over the modeled field.

By default, KRISP-U now learns an empirical kernel prior from the current
measured dataset before every recommendation. It estimates a variogram summary,
scores Matern/RBF/RationalQuadratic candidates with cross-validation, likelihood,
and uncertainty coverage, then fits the selected GPR kernel before scoring the
next point. Pass `GprConfig(adaptive_kernel=False)` to use the historical fixed
`Matern(2.5)` kernel.

## Install

```bash
python -m pip install -e .
```

For plotting and development tools:

```bash
python -m pip install -e ".[plot,dev]"
```

## One-Shot Research Workflow

Give KRISP-U a CSV of completed measurements and ask for the next point or
ranked set of points to measure. If the CSV contains rows with blank target
values, KRISP-U treats those rows as the allowed candidate pool.

```bash
krispu-recommend examples/researcher_measurements.csv \
  --target response \
  --features x1 x2 \
  --n-recommendations 3 \
  --output krispu_recommendations.csv
```

The output CSV contains `rank`, feature values, acquisition score, predicted
mean, and predicted uncertainty. After measuring a recommended point, append the
new response to the dataset and run the command again.

The same workflow is available from Python:

```python
import numpy as np
from krispu import DiscreteCandidateSpace, recommend_next

X_measured = np.array([[-4.0, 14.0], [-2.0, 4.0], [2.5, 12.0], [5.0, 4.0]])
y_measured = np.array([3.958293, 30.602112, 24.129964, 15.829732])
candidates = np.array([[-3.1416, 12.275], [3.1416, 2.275], [9.4248, 2.475]])

space = DiscreteCandidateSpace(candidates, names=("x1", "x2"))
recommendations = recommend_next(
    X_measured,
    y_measured,
    space=space,
    candidates=candidates,
    n_recommendations=2,
)

print(recommendations.to_records())
print(recommendations.selected_kernel_family)
```

For continuous spaces with no preset candidates, pass bounds:

```bash
krispu-recommend measurements.csv \
  --target response \
  --features temperature pressure \
  --bounds temperature:20:80 pressure:1:5 \
  --n-recommendations 5
```

## Sequential Toy Runs

Toy runs use the same GPR sampler, but the script evaluates the toy response
function automatically after each selected point:

```python
from krispu import KrispUOptimizer, get_dataset

dataset = get_dataset("branin")
initial_X = dataset.initial_design(n=6, random_state=7)
initial_y = dataset.evaluate(initial_X)

sampler = KrispUOptimizer(
    dataset.space(),
    objective=dataset.objective,
    acquisition="kld",
    random_state=7,
)
result = sampler.run(
    dataset.evaluate,
    initial_X=initial_X,
    initial_y=initial_y,
    n_iterations=12,
)

print(result.X)
```

## Benchmark Evidence

KRISP-U includes matched-budget baselines for random sampling, naive grid
sampling, and Latin-hypercube/Sobol space-filling sampling.

```python
from krispu import run_benchmark

result = run_benchmark(
    "branin",
    methods=("krispu", "random", "grid", "lhs"),
    budget=20,
    n_initial=None,
    n_trials=10,
    random_state=11,
    initial_design="hull",
    tolerance=1.0,
)

print(result.summary())
print(result.compare_to_baseline("krispu", "random", metric="field_nrmse"))
```

For 2D continuous domains, `initial_design="hull"` starts from the four domain
corners plus one random interior measurement. Benchmarks then mimic the research
workflow: each method receives the same starting database, proposes one new
measurement at a time, and is scored against a hidden dense validation field.

The primary benchmark metrics are final field RMSE/MAE, normalized RMSE/MAE,
and field R2. Normalized RMSE is `field_rmse / response_range`, so it remains
comparable across toy fields with different response scales. R2 reports field
variance explained: `1.0` is perfect reconstruction, `0.0` matches a domain-mean
predictor, and negative values are worse than the domain mean. Additional
diagnostics include p95 absolute error, worst-region absolute error, 95%
uncertainty coverage, mean predictive uncertainty, uncertainty reduction, and
R2/NRMSE area under the measured-point learning curve. The paired comparison
reports `krispu - baseline`, so a negative `field_nrmse` difference means KRISP-U
had lower normalized error; a positive `field_r2` difference means KRISP-U
explained more field variance.

The visual benchmark script also tracks sample efficiency to `field_r2 >= 0.90`.
Each method/seed trial stops as soon as its hidden-field R2 crosses the target,
with 96 measured points retained as the visual-suite hard cap for unreached trials. The
threshold table reports the first measured-point count crossing the target; if a
run does not reach the threshold by the cap, the table marks it as unreached and
the bar plot caps it at the maximum budget. With early stopping enabled, final
metrics in `benchmark_summary.csv` are measured at each trace's stop point or
cap, not at a shared final budget.

The main visual benchmark compares `krispu_fixed`, `krispu_adaptive`, random,
grid, and LHS. `krispu_adaptive` repeats kernel-prior optimization after each new
measurement; `krispu_fixed` preserves the historical fixed-kernel behavior.

`final_regret_mean` is retained only as a secondary toy-function diagnostic. For
minimization it is `max(best_observed_y - known_best_y, 0)` at the final budget;
for maximization it is `max(known_best_y - best_observed_y, 0)`. It does not
measure field matching.

To regenerate benchmark figures and tables:

```bash
python examples/generate_benchmark_visuals.py
```

The script writes figures and tables to `benchmark_outputs/`, including
`field_nrmse_summary.png`, `field_nrmse_vs_n.png`, `field_r2_summary.png`,
`field_r2_vs_n.png`, `field_p95_abs_error_vs_n.png`,
`field_max_abs_error_vs_n.png`, `field_coverage_95_vs_n.png`,
`mean_uncertainty_vs_n.png`, `benchmark_domains_2x3.png`,
`benchmark_n_sweep_summary.csv`, `r2_threshold_summary.csv`,
`r2_90_points_to_threshold.png`, and
`krispu_normalized_field_error_comparisons.csv`. The `*_vs_n` plots are built
from one sequential trajectory per seed, not independent reruns at each `n`.

The main 2D visual benchmark suite uses continuous response fields only:
Branin, six-hump camel, Ackley 2D, quadratic bowl, anisotropic ridge, and a
continuous Gaussian mixture.

To generate prediction/uncertainty GIFs that show KRISP-U stepping through a
2D domain:

```bash
python examples/generate_prediction_uncertainty_gifs.py
```

The GIFs show GPR prediction on the left, acquisition score on the right, and
the next selected measurement in each frame. By default the GIF script uses the
KLD information-gain acquisition; pass `--acquisition uncertainty` to visualize
plain predictive standard deviation instead.

To test the sparse "convex hull plus one interior point" start:

```bash
python examples/run_hull_start_experiments.py
```

For 2D domains this starts from the four corners plus one random interior
measurement, then records KRISP-U-selected follow-up points in
`benchmark_outputs/hull_start/`.

To generate a separate higher-dimensional bar-chart comparison against random,
grid, and LHS sampling:

```bash
python examples/generate_high_dimensional_benchmark_bars.py
```

The chart and summary tables are written to
`benchmark_outputs/high_dimensional/`.

To make GIFs for those hull-start cases:

```bash
python examples/generate_prediction_uncertainty_gifs.py \
  --initial-design hull \
  --datasets quadratic_bowl_2d branin six_hump_camel ackley_2d gaussian_mixture_2d \
  --max-iterations 24 \
  --random-state 151 \
  --output-dir benchmark_outputs/hull_start/gifs
```

## Included Toy Fields

Continuous examples include Forrester, Branin, Himmelblau, six-hump camel,
Ackley 2D, Goldstein-Price, Rosenbrock 2D, anisotropic ridge, Gaussian mixture,
quadratic bowl, noisy smooth surface, Hartmann 3D/6D, and a synthetic 5D
additive function.

Preset-candidate examples include irregular Branin candidates, sparse Gaussian
mixture candidates, and coarse anisotropic ridge candidates.

## Plotting

Install the plotting extra and use `krispu.plotting` for prediction maps,
uncertainty maps, acquisition maps, benchmark comparisons, discrete candidate
scatters, and pairwise n-D slices.

```python
from krispu.plotting import plot_2d_prediction, plot_2d_uncertainty

plot_2d_prediction(sampler)
plot_2d_uncertainty(sampler)
```

## Development Checks

```bash
ruff check .
black --check .
pytest
```

## Licensing and Citation

This work is licensed under a Creative Commons Attribution-ShareAlike 4.0
International License.

Cite as:

Matthew Burnett and Austin Downey. Kriging with Iterative Spatial Prediction of
Uncertainty (KRISP-U) Algorithm. GitHub.
https://github.com/ARTS-Laboratory/KRISP-U

```bibtex
@Misc{BurnettKrigingIterativeSpatial,
  author       = {Matthew Burnett and Austin Downey},
  howpublished = {GitHub},
  title        = {Kriging with Iterative Spatial Prediction of Uncertainty (KRISP-U) Algorithm},
  url          = {https://github.com/ARTS-Laboratory/KRISP-U},
}
```
