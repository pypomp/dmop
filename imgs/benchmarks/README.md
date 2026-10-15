# Cross-model benchmark figures and table

The source is [../../code/benchmarks/](../../code/benchmarks/README.md).
All declared final batches and separate Gaussian particle-count tuning are
complete. To regenerate the publication outputs from the repository root, run:

```sh
python code/benchmarks/report.py
```

The command requires all four methods on 20 starts for each of the oscillator,
linear Gaussian, SPX and Daphnia models, plus the archived 100-start Dhaka
comparison. It rejects incomplete runs or incompatible starting vectors.
The Gaussian panels use the common IFAD/IF2/DS19 particle count selected on
separate tuning starts. No fitting or parameter selection occurs here.

Generated files:

| File | Contents |
|---|---|
| `final_likelihoods.pdf`, `.png` | Final estimates, boxplots and failure markers; Dhaka axis restricted to [-3840,-3740] |
| `final_likelihoods_full.pdf`, `.png` | Uncropped version, including Dhaka estimates below -3840 |
| `optimization.pdf`, `.png` | Median progress and 10th–100th percentile bands over their full range |
| `optimization_detail.pdf`, `.png` | The same progress near the final estimates, on separate linear axes |
| `results_table.tex`, `summary.csv` | Central table of likelihoods, spread, Monte Carlo errors, time and failures |
| `final_results.csv`, `progress_summary.csv` | Values used in the figures and table |
| `provenance.json`, `plot_ranges.json` | Input hashes, particle count, reference definitions and axis limits |

[figures.tex](figures.tex) supplies the figure and table environments and
captions included in [the SI](../../si.tex). The final panels show all five
models, including all 20 Daphnia starts and the archived 100 Dhaka starts.
The displayed Dhaka final and progress detail panels use [-3840,-3740].
Counts below the final plot's axis are recorded in `plot_ranges.json` and
shown on the plot. Every estimate remains in the table and saved data.

For a development preview of completed models, pass `--models`, an explicit
`--gaussian-particles` value if tuning is unfinished, and a separate
`--output` directory. Keep such partial previews outside this publication
directory. The Dhaka score-versus-numerical-SAEM A/B study is a separate
experiment in [../../ditlevsen/](../../ditlevsen/README.md).
[saem_ab_figures.tex](saem_ab_figures.tex) supplies its SI captions and
references the corrected `saem_ab_scaled/report/` outputs. All 40 paired
comparisons are complete, audited and included in the SI through this fragment.
