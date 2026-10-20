# Making Figures

The three tutorial animations recreate the original kriging figure's broad
prediction trend and fixed color scales in a square tutorial domain:

```text
python make_gifs.py
```

This writes `Gif.gif` (uncertainty only, with a fixed 0--1 colorbar),
`Prediction.gif` (mean prediction only, with fixed 0--1 color scaling), and
`Prediction_Uncertainty.gif` (both views rendered directly in one figure).
The parallel `*_original_labels.gif` files contain the identical frames with
the original `Peak`, `width`, and `# to failure (output)` labels.

The v0.3 jackknife kriging slide figure is stored at
`versions/v0.3.0/slide_figures/jackknife_kriging.gif`, with its frame trace in
the neighboring CSV file. Regenerate it from the v0.3 environment with:

```text
python examples/generate_jackknife_kriging_gif.py
```

The smooth sequential prediction/uncertainty figure is stored at
`versions/v0.3.0/slide_figures/smooth_toy_prediction_uncertainty.gif`.
Its companion uncertainty/$R^2$ plot is stored at
`versions/v0.3.0/slide_figures/smooth_toy_uncertainty_r2.png`.



