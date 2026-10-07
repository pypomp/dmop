# Dhaka comparison tables and figures (SI Section S11)

From the repository root:

```sh
python code/competitor_results.py
```

Dependencies: NumPy, pandas, and Matplotlib (tested with 2.4.2, 3.0.0, and
3.10.8). The script reads tracked CSVs only; no GPU, archived checkpoints,
or sibling repository is required. It writes the two LaTeX table fragments,
CSV summaries, and PDF/PNG versions of the two figures in this directory.

`summary.csv` records each result's source directory. `paired_summary.csv`
uses each start's own independently evaluated IF2 checkpoint. The generator
checks the 100 unique start identities, final evaluation success and effort
(5,000 particles, 36 replicates), and agreement of paired differences with
the existing objective-mismatch exports. All 100 runs contribute, including
early optimizer stops; plot limits do not discard low-likelihood points.

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
