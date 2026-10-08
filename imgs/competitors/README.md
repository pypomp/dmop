# Dhaka comparison tables and figures (SI Section S11)

From the repository root:

```sh
python code/competitor_results.py
```

Dependencies: NumPy and pandas (tested with 2.4.2 and 3.0.0).
The script reads tracked CSVs only; no GPU, archived checkpoints,
or sibling repository is required. It writes the two LaTeX table fragments
and CSV summaries in this directory.

The SI includes the final-likelihood, full-optimization, continuation, and
parameter-density PDFs in
[corenflos100_ditlevsen1000/](corenflos100_ditlevsen1000/README.md).
Both CTDD21 variants use 100 fitting particles;
both DS19 variants use 1,000. These figures were produced by
`corenflos/corenflos/settings_plots.py` using the established `warm_plots.py`
workflow. All 100 starts contribute to summaries; the displayed limits are coordinate zooms only.
The tables retain all tested configurations.

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

Fitting and evaluation are described in the
[shared reproduction guide](../../code/competitor_reproduction.md), with
source maps in the [CTDD21](../../corenflos/README.md) and
[DS19](../../ditlevsen/README.md) guides. Bulk-result restoration is
described in [artifacts/README.md](../../artifacts/README.md).
The frozen per-run `configuration.json` files
and the 1,000-particle `manifest.json` are authoritative for settings.

## Mixed particle settings (October 7, 2026)

The [corenflos100_ditlevsen1000](corenflos100_ditlevsen1000/) set used in the SI has
fixed particle counts for each method across global and IF2-initialized
searches. These are separate initialization experiments.

| Method | Fitting particles | Initial learning rate |
|---|---:|---:|
| CTDD21 | 100 | 0.02 |
| CTDD21 + IF2 warm start | 100 | 0.0002 |
| DS19 | 1,000 | 0.1 |
| DS19 + IF2 warm start | 1,000 | 0.0005 |

Each arm includes all 100 starts. Final likelihoods use independent Euler-20
evaluation with 5,000 particles and 36 replicates; trajectory points use
5,000 particles and one replicate. IF2 checkpoint and current IFAD-0.97
references are unchanged. Particle counts in labels refer to fitting, not
evaluation. Standalone fits have an 800-second budget and select outputs
around 700 seconds under their original rules. Warm fits have 850.657 seconds
after the 92.160-second IF2 stage, with a nominal 942.817-second total endpoint.
The curves show optimizer trajectories, not necessarily their selected outputs.

Each figure is saved as PNG and PDF:

| Filename stem | View |
|---|---|
| `likelihood_if2warm_comparison_r20` | Final-output overview, -4800 to -3735 |
| `likelihood_all_outputs_r20` | All final outputs, no likelihood cutoff |
| `optimization_if2warm_elapsed_full_r20` | All six method traces, -4800 to -3735 |
| `optimization_if2warm_elapsed_r20` | IF2 and continuation zoom, -3820 to -3735 |
| `optimization_all_values_r20` | Complete trace summaries, no likelihood cutoff |
| `parameter_if2warm_comparison_r20` | Parameter densities |
| `objective_mismatch_if2warm_medians_r20` | Median paired training and Euler changes |
| `objective_mismatch_if2warm_paired_r20` | All 100 paired changes per warm-start method |

The overview limits are coordinate zooms only: no starts are discarded before
computing densities, boxplots, or summaries. The uncropped plots show the tails
and initial trace segments outside those windows. Trace lines are medians;
bands extend from the 10th percentile to the maximum. Mismatch diamonds show
the marginal medians. This set omits the historical -3744.17 reference line
used by earlier plots.

`settings.csv` identifies each source, particle count, learning rate, budget,
median likelihood, and update count; `settings.json` preserves complete fitting
configurations. `final_all_runs.csv`, `optimization_summary.csv`, and
`objective_mismatch.csv` preserve the numerical inputs to the comparison.

After following the [environment setup](../../code/competitor_reproduction.md),
regenerate from `dmop/corenflos` with your active Python interpreter:

```sh
PYTHONPATH=.:../ditlevsen:../../pypomp JAX_PLATFORMS=cpu JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1 \
  python -m corenflos.settings_plots
```

This uses the established `warm_plots.py` workflow and existing local trace
CSVs. A fresh clone requires bulk-trace restoration as described in the
artifact archive instructions above; the compact exported summaries and
figures remain tracked.
