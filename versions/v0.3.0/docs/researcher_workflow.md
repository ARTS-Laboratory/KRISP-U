# Researcher workflow

The active package is `src/krispu`. Import reusable workflows from
`krispu.api` or use the CSV command:

```bash
krispu-recommend measurements.csv \
  --target response \
  --features x y \
  --bounds x:-1:1 y:-1:1 \
  --n-recommendations 3 \
  --output outputs/researcher/recommendations.csv
```

Rows with a finite target are measured observations. Blank-target rows are
optional user-supplied candidates. If none are supplied, the command creates a
Latin-hypercube candidate pool inside the declared domain.

For evaluation, use a YAML suite profile rather than algorithm flags:

```bash
python -m evaluation.runners.suite --config configs/suites/development.yaml
```

The runner resolves defaults, rejects unknown keys, records the resolved YAML,
and writes to the stable overwrite-only directory
`outputs/<experiment_name>/`. It does not expose the algorithm's hyperparameters
through a large command-line interface.

Keep active source changes in `src`, stable evaluation machinery in
`evaluation`, and experiments that are not yet promoted in `scratch`. Never
import scratch or generated outputs from either active source or evaluation.
