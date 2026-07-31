# KRISP-U

Kriging with Iterative Spatial Prediction of Uncertainty (KRISP-U) is research
software for reconstructing a field and selecting informative measurements.
This repository preserves two independently runnable scientific
implementations. They are versioned methods, not interchangeable directory
layouts, and neither implementation silently falls back to the other.

## Versions

| Version | Implementation | Primary interface |
| --- | --- | --- |
| [v0.1.0](versions/v0.1.0/README.md) | PyKrige with leave-one-out resampling, whole-field divergence, and an interpolated uncertainty map | `KRISPU` class |
| [v0.2.0](versions/v0.2.0/README.md) | Gaussian Process Regression active field reconstruction and candidate recommendation | `krispu` package and `krispu-recommend` |

The original online v0.1.0 state is also preserved by the Git tag `v0.1.0`.
The complete merged repository is tagged `v0.2.0`. Source provenance and
checksums are recorded in [`versions/manifest.yaml`](versions/manifest.yaml).

## Installation

Install only the version required for a particular analysis, preferably in a
dedicated virtual environment.

```bash
cd versions/v0.1.0
python -m pip install -e .
```

```bash
cd versions/v0.2.0
python -m pip install -e ".[plot]"
```

See each version's README for its workflow and examples. Presentation and
tutorial materials are under [`docs/tutorial`](docs/tutorial), and the
repository QR code is under [`docs/media`](docs/media).

Generated `benchmark_outputs` and `recommendation_outputs` directories are
intentionally not version-controlled. Raw example inputs and curated figures
are retained with the version that uses them.

## Licensing and citation

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

Cite as:

Matthew Burnett and Austin Downey. *Kriging with Iterative Spatial Prediction
of Uncertainty (KRISP-U) Algorithm*. GitHub.
https://github.com/ARTS-Laboratory/KRISP-U

```bibtex
@Misc{BurnettKrigingIterativeSpatial,
  author       = {Matthew Burnett and Austin Downey},
  howpublished = {GitHub},
  title        = {Kriging with Iterative Spatial Prediction of Uncertainty {(KRISP-U)} Algorithm},
  groups       = {{ARTS-Lab}},
  note         = {Accessed: 20xx-xx-xx},
  url          = {https://github.com/ARTS-Laboratory/KRISP-U},
}
```

<p align="center">
<img src="docs/media/QR-code.png" alt="KRISP-U repository QR code" width="200"/>
</p>

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg



