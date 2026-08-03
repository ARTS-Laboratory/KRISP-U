# Architecture

`src/krispu` contains reusable scientific functionality: domains, observations,
kernels, surrogates, jackknife/uncertainty calculations, acquisition scores,
and result objects. `krispu.api` is the small public surface:
`fit_reconstruction`, `evaluate_uncertainty`, and `recommend_next_point`.
Benchmark orchestration is not part of that API.

`evaluation` contains stable research machinery. Fields are under
`evaluation/fields`, candidate-order methods under `evaluation/methods`,
metrics under `evaluation/metrics`, and orchestration under
`evaluation/runners`. The sequential runner is evaluation code because it
coordinates benchmark methods and metrics. Plotting consumes completed state
or result objects; it does not fit models or make acquisition decisions.

`scratch` is disposable and may not be imported. `outputs` contains generated
suite results and is ignored by version control.

The dependency direction is strict:

```text
evaluation -> krispu
krispu -X-> evaluation
krispu -X-> scratch
krispu -X-> outputs
```

The regression suite imports every `krispu` module and inspects its imports to
keep this boundary explicit.
