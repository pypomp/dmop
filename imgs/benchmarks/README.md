# Cross-model benchmark figures and table

The source is [../../code/benchmarks/](../../code/benchmarks/README.md).
From the repository root, after the declared final batches and separate
Gaussian particle-count tuning have completed, run:

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
| `final_likelihoods.pdf`, `.png` | All final estimates, boxplots and failure markers |
| `optimization.pdf`, `.png` | Median progress and 10th–100th percentile bands over their full range |
| `optimization_detail.pdf`, `.png` | The same progress near the final estimates, on separate linear axes |
| `results_table.tex`, `summary.csv` | Central table of likelihoods, spread, Monte Carlo errors, time and failures |
| `final_results.csv`, `progress_summary.csv` | Values used in the figures and table |
| `provenance.json`, `plot_ranges.json` | Input hashes, particle count, reference definitions and axis limits |

[figures.tex](figures.tex) supplies the SI figure and table environments and
captions. Include it from `si.tex` only after all generated artifacts have
been reviewed. The source fragment alone does not indicate that the jobs
or manuscript integration are complete.

For a development preview of completed models, pass `--models`, an explicit
`--gaussian-particles` value if tuning is unfinished, and a separate
`--output` directory. Keep such partial previews outside this publication
directory. The Dhaka score-versus-numerical-SAEM A/B study is a separate
experiment in [../../ditlevsen/](../../ditlevsen/README.md).
