# KRISP-U v0.1.0

This directory preserves the original PyKrige implementation of Kriging with
Iterative Spatial Prediction of Uncertainty. The scientific source and example
data are unchanged from Git commit
`8e7530fee57cb0a511647d2bf9b5a3dce975f2a8`.

The workflow fits a two-dimensional kriging field, removes eligible measured
points in turn, compares each leave-one-out field with the full-data field,
interpolates the resulting pointwise divergence values, and chooses a new point
from that uncertainty map.

## Install

From this directory:

```bash
python -m pip install -e .
```

The historical public modules remain top-level modules:

```python
from KRISPU import KRISPU
from Utilities import JSD, KLD, MSE
```

## Run the historical example

The example uses paths relative to its original `source` directory:

```bash
cd source
python example.py
```

`README.original.md` preserves the former repository-level README, while
`source/README.md` preserves the original source notes. Generated example
figures are retained because they document the historical analysis outputs.

This version is preserved for reproducibility. Its equations, handling of
invalid divergence results, interpolation, and point-selection behavior have
not been revised during repository consolidation.
