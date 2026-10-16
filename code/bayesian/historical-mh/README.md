# Historical Metropolis traces

These four saved 1,500-by-21 arrays reproduce the drifting MH chains in
`imgs/pmcmc/mh.png`, including the trend chain ending at -1.219. They were
recovered from diffPomp commit `8d2fa90`; provenance and SHA-256 hashes are
in `provenance.json`. The manuscript image matches the image in diffPomp
HEAD byte for byte. This archive complements the empirical-prior NUTS
archive in the parent directory.

The notebook at the first commit containing these draws saves an active
NUTS calculation. Its MH alternative remains commented out. No active MH
sampling cell was found in the available history of `cholera.ipynb`.
Consequently, the code below is evidence of the intended setup, not a
complete execution record:

- A diagonal Gaussian random walk with estimation-scale SD equal to one
  hundredth of each search-range width. For `100*beta_trend`, this is .01.
- 10,000 particles, 1,500 iterations and no proposal adaptation.
- An IF2-fitted KDE prior, separately constructed for each chain.
- The commented target is `-mop_mean - dprior`; both terms divide by the
  600 observation count. `key=None` introduces tracing-time randomness.

The original KDE clouds, actual initial states, acceptance diagnostics and
tracing-time seeds are missing. The first saved draw is not necessarily
the initial state. A fresh exact replay cannot be claimed from these files.
`notebook_cells.py`, `filtering.py` and `pomps.py` are historical source for
inspection; they are not imported by the current Pypomp experiments.

`compare_traces.py` compares these saved trend draws with the first 1,500
iterations of current PMMH on common linear axes, using iterations501–1500
for both density panels. It freezes the plotted arrays and records source
hashes under `comparison/`. These runs differ in target, prior, starts and
proposal; the plot does not isolate a prior effect or establish a speedup.

Run from the repository root with NumPy, SciPy and Matplotlib installed:

```sh
python code/bayesian/historical-mh/compare_traces.py
```

The new control in
`../corrected/results/historical-proposal-control-v1` restores the historical
proposal scale without adaptation, keeping the current Pypomp likelihood,
proper prior and initial states fixed. This tests the proposal difference;
it does not recreate the missing historical KDE or reinstate tempering.
