# Figure 1A: `imgs/095/mop.png`

From the repository root, redraw the figure after editing labels in `plot.py`:

```sh
python code/fig1a/plot.py
```

This takes only NumPy and Matplotlib and uses the checked-in `curves.csv`.
The default destination is `imgs/095/mop.png`; `--output /tmp/mop.png` writes
a preview. Paths are relative to the script, so the command also works from
another directory. The alpha=1 legend now reads
**Alpha=1 (similar to Poyiadjis, 2011)**; the alpha=0 legend reads
**Alpha=0 (similar to Naesseth, 2018)**.

## Where the figure came from

The original is in [`hetankevin/diffPomp`, `cholera_mop.ipynb`, revision
`ab31c911ae223aa91eb4e216d7d33710103050c3`](https://github.com/hetankevin/diffPomp/blob/ab31c911ae223aa91eb4e216d7d33710103050c3/cholera_mop.ipynb).
Its zero-based cells 2, 3, and 4 define the model/filter inputs, evaluate
the curves, and draw the figure. The PNG embedded in cell 4 has SHA-256
`f00e9fa2bd73c66a1666818dfd41c813398648287bc56f52cc69e7472cac37ab`,
identical to the manuscript's `mop.png` before the October 7, 2026 edit.
The local copy was found in `haitiRes/diffPomp`; its working notebook contains
merge-conflict markers, so the extraction used the intact Git revision.

This directory contains everything needed to recompute the figure:

- `legacy_model.py`: the used functions from that revision's Gaussian
  `pomps.py`, with unrelated imports and inactive alternatives removed.
- `legacy_filters.py`: the two-pass particle-filter/MOP functions from cell 2
  and `normalize_weights` from `resampling.py`.
- `data/{dacca,covars,covart}.csv`: the original data and covariates from the
  same revision, including the original time-index interpolation inputs.
- `generate.py`: the parameter settings and grid calculation from cells 2–3.
- `figure.mplstyle` and `plot.py`: the original style settings and plotting
  logic, with the requested legend qualification.
- `curves.csv` and `curves.json`: recomputed likelihood values and run metadata.

The legacy model is intentionally separate from the newer fitting code.
In particular, the old checkout's current `pomps.py` has been changed to
Gamma noise; that is **not** the Gaussian model used for this figure.

## Recompute the curves

With JAX, NumPy, and pandas installed, run:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false python code/fig1a/generate.py
python code/fig1a/plot.py
```

The saved October 7 run used JAX 0.9.0.1 on an RTX 3090, NumPy 2.4.2,
pandas 3.0.0, and Matplotlib 3.10.8 for plotting. GPU computation can take several
minutes, especially with deterministic settings; CPU execution can be much slower. `--particles` and `--output`
allow separate diagnostic runs; the manuscript setting is 10,000 particles.
All other parameters are explicit in `generate.py` and `legacy_model.py`.

The baseline recovery rate is 18. The notebook's
`linspace(17.5, 18.5, 101).round(1)` has only 11 distinct values. We compute
each once because every call has the same seed. Both filters deliberately
reset `PRNGKey(0)` at each month, as in the notebook. The generator fixes
float32 arithmetic and the original, non-partitionable Threefry stream.
It also enables deterministic GPU operations and disables GPU autotuning: an
uncontrolled GPU rerun changed the float32 particle genealogy despite fixed
seeds. The deterministic replay matched the first three saved grid points;
a complete replay with those settings has not been checked.
This is a reconstruction of that illustration, not a change to the current
benchmark's random-number conventions.

The dashed curve labeled "True Log-Likelihood" in the original figure is
itself a particle-filter estimate. The three smooth curves hold the baseline
resampling measurements fixed, using alpha values 1, 0.97, and 0.
CSV entries are log likelihoods, whereas the legacy filter functions return
negative log likelihoods.

The original notebook saved its PNG but did not export its numeric curves.
The committed curves were recomputed from the recovered source, not digitized
from the image. JAX/compiler/device changes and float32 resampling mean a
recomputation need not reproduce the old PNG pixel for pixel. Future label
edits using `plot.py` preserve the committed curve values exactly.
