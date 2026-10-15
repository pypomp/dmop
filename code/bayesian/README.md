# Archived Bayesian calculation

This directory documents the original empirical-prior NUTS calculation in
main-text Figure 6 and SI S8. It preserves the saved draws and the sampling
source from [diffPomp revision 65e64d3](https://github.com/hetankevin/diffPomp/tree/65e64d3e24d7d82839b413067c30b4224dee43c9).
These are historical runs, not a new Bayesian experiment.

| File | Purpose |
|---|---|
| `source/sampling_cells.py` | Verbatim code cells 0–7 of the historical `cholera.ipynb`, with cell markers |
| `source/{filtering,pomps,optim,resampling}.py` | Unmodified imported modules at that revision; archived for inspection, not imported by the diagnostic script |
| `data/theta_0.npy` through `theta_3.npy` | Four original 1,500-by-21 chains, byte-for-byte verified against the Git objects |
| `provenance.json` | Source revision, hashes, observation count and known missing artifacts |
| `diagnostics.py` | Recompute diagnostics from the saved draws without fitting |
| `results/` | Full numeric diagnostics, SI table and audit record |

The later `cholera_npeb.ipynb` plots these same files. Its saved
`imgs/pmcmc/nuts_eb.png` has the same SHA-256 as the manuscript's image.
The notebook was subsequently changed to alpha .97, but the revision
which saved these chains specifies alpha **.95**. We report that historical
source instead of silently substituting the later setting.

## Actual calculation

- Each chain constructs its own 19-dimensional joint Gaussian KDE from an
  independently fitted IF2 swarm. IF2 uses 40 iterations, 10,000 particles,
  initial perturbation SD .02 and cooling factor .95 each iteration. The
  highest monitored-likelihood swarm supplies both the KDE and the starting
  mean. Scott's bandwidth factor is `10000**(-1/23)`; kernel covariance is
  this factor squared times the swarm sample covariance.
- Positive parameters use logs, the trend uses `100*beta_trend`, and
  seasonal coefficients are unchanged. Stored columns 2 and 5 are fixed at
  minus/plus infinity, representing rho=0 and c=1. Column 4, log omega,
  varies in the KDE although the simulator replaces constant omega by its
  seasonal environmental term. Initial state proportions are fixed.
- The supplied log potential is `(particle_log_objective + log_KDE)/600`.
  Both likelihood and prior are tempered; this is not the ordinary posterior.
  The MOP routine uses 1,000 particles and systematic resampling. Its numeric
  forward value and its stop-gradient derivative must be distinguished.
- `key=None` constructs a NumPy-generated seed inside the JIT function;
  resampling also obtains its NumPy offset during tracing. Compiled calls
  reuse these constants. Genealogies change with candidate parameters.
  No random-number state is explicitly updated by a pseudo-marginal kernel.
  Original tracing-time seeds and package versions were not recorded.
- TFP JAX NoUTurnSampler uses fixed step size .05, tree depth at most 5,
  energy threshold 1,000, no adaptation and chain seeds 0–3. It saves 1,500
  draws, with 500 subsequently discarded in the plots. The library's NUTS
  transition uses the supplied potential; no additional likelihood correction
  is implemented. An invariant-distribution justification is not supplied
  by this archive audit.

The old “6.3% of the MLE” statement came from
`.02 * .95**40 * sqrt(600) = .0629578`. This is a heuristic accumulation
of final-iteration perturbations on the transformed scale. It is neither
the measured swarm SD nor the KDE SD, and the selected swarm can occur
before iteration 40. The statement has been removed.

## Diagnostics and limits

Run with NumPy, pandas and **ArviZ 0.22.0** installed:

```sh
python code/bayesian/diagnostics.py
```

The script verifies the archived hashes, excludes only the two fixed
coordinates, and retains all 1,000 post-discard draws in every run.
Rank-normalized split R-hat compares the four runs descriptively: their
priors differ, so it is not a convergence test for one common target.
ESS and mean MCSE are calculated within each chain; the SI gives the
minimum ESS and maximum MCSE, with every individual result saved in CSV.
The estimates require stationarity and do not certify posterior accuracy.

The notebook used `trace_fn=None` and did not save the KDE clouds, kernel
results or timings. Therefore divergence counts, tree-depth saturation,
ESS per second and original prior overlays cannot be reconstructed from
the archived draws. Do not replace these with zeros or new-run values.
The earlier 1,000-draw files in diffPomp are different runs and are not
pooled with these 1,500-draw chains.
