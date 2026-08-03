# KRISP-U v0.3.0

KRISP-U reconstructs a response field with a response-standardized Matérn
Gaussian process and ranks candidate measurements with support-adjusted buffered-jackknife
uncertainty. The numerical core is the reusable code in `src/krispu`.

Install from this directory:

```bash
python -m pip install -e ".[dev,plot]"
```

On Windows, the setup script creates or repairs the local virtual environment:

```powershell
.\setup.ps1
```

User-facing workflows are intentionally small:

```python
import numpy as np

from krispu.api import recommend_next_point
from krispu.domains import ContinuousDomain
from krispu.observations import ObservationSet

domain = ContinuousDomain([[-1, 1], [-1, 1]], names=["x", "y"])
observations = ObservationSet(
    X=np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]),
    y=np.array([0.0, 1.0, 1.0, 0.0]),
)
result = recommend_next_point(domain, observations)
print(result.recommendations[0].x)
```

The active algorithm source is only `src/krispu`. Stable benchmark fields,
methods, runners, metrics, figures, and reports live under `evaluation` and
may import `krispu`; the algorithm source never imports `evaluation`,
`benchmarks`, `scratch`, or `outputs`. `scratch` is disposable: useful work
must be promoted into `src`, `evaluation`, or `tests`.

Run a reproducible suite with one minimal command:

```bash
python -m evaluation.runners.suite --config configs/suites/development.yaml
```

Run every Pass 3 suite in order with:

```bash
python -m evaluation.runners.run_all_suites
```

From the repository root, use:

```powershell
python .\versions\v0.3.0\run_all.py
```

The wrapper writes each suite under `outputs/<suite_name>/` and updates
`outputs/all_suites_manifest.yaml` after every suite. Use `--fail-fast` to stop
on the first failure or repeat `--suite NAME` to run a selected subset.

YAML profiles are validated before execution. Unknown keys are errors, and the
resolved profile is saved as `outputs/development/config_resolved.yaml`.
Outputs are overwrite-only at `outputs/<suite_name>/`; rerunning a suite
replaces that exact suite directory and never creates timestamped run folders.

The default `summary` profile writes scalar tables, kernel tables, the four
official field figures, the four official global figures, a manifest, and a
report. `diagnostic` adds checkpoint arrays and complete candidate-score
tables. `debug` additionally retains intermediate arrays and animation frames;
temporary frames are removed after GIF construction in other modes.

See `docs/architecture.md`, `docs/researcher_workflow.md`, and
`docs/output_schema.md` for the source/evaluation boundary and output schema.
