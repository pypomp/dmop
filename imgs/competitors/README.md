# Dhaka comparison tables and figures (SI Section S11)

From the repository root:

```sh
python code/competitor_results.py
```

Dependencies: NumPy and pandas (tested with 2.4.2 and 3.0.0).
The script reads tracked CSVs only; no GPU, archived checkpoints,
or sibling repository is required. It writes the two LaTeX table fragments
and CSV summaries in this directory.

The SI uses the previous agent's existing likelihood, full optimization,
focused continuation, and parameter-density PNGs directly from:

- `corenflos/results/if2warm_ifad097_comparison/figures/`
- `corenflos/results/particle_increase_j1000_final_100/figures/`

These figures were produced by `corenflos/corenflos/warm_plots.py`; they
have not been redrawn or modified for the SI. The replacement plots
created during manuscript integration were removed at the user's request.
The original plots retain their display cutoffs, stated in the SI captions.
In particular, the raincloud distributions exclude values below -4300
before summarizing, whereas the tables here include all 100 runs.

`summary.csv` records each result's source directory. `paired_summary.csv`
uses each start's own independently evaluated IF2 checkpoint. The generator
checks the 100 unique start identities, final evaluation success and effort
(5,000 particles, 36 replicates), and agreement of paired differences with
the existing objective-mismatch exports. All 100 runs contribute to the
tables, including early optimizer stops.

The IFAD-0.97 reference comes from the current tracked
`ditlevsen/results/reference/manuscript_likelihood.csv` (comparable effort),
which agrees with `imgs/precise_table.tex`: median -3744.40 and maximum
-3743.68. Older experiment README prose quotes superseded reference values.
The IF2 checkpoint is the intermediate 175-iteration warm start, not the
full-budget IF2 comparison in the main manuscript.

Fitting and evaluation reproduction is documented in `../../corenflos/README.md`
and `../../ditlevsen/README.md`. Bulk-result restoration is described in
`../../artifacts/README.md`. The frozen per-run `configuration.json` files
and the 1,000-particle `manifest.json` are authoritative for settings.
