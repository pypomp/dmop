# Identity-metric NUTS and diagonal PMMH

The completed comparison is in SI Section S8.4. Four NUTS chains have 500
warmup and 1,000 retained iterations. The metric is identity on the original
estimation coordinates; only step sizes adapt during warmup. Diagonal PMMH
uses fixed standard deviation 0.02 in prior-standardized coordinates and
has no adaptation. Both methods use the same target and starting points.
The comparison uses iterations 501--1,500 for both.

NUTS has zero divergences but poor mixing: trend R-hat 1.380 and bulk ESS
8.93, versus 2.395 and 4.93 for diagonal PMMH. Worst R-hat across coordinates
is 3.491 versus 2.580. Neither run supports convergence. All retained NUTS
trajectories reach depth 5. The plotted comparisons are per iteration;
recorded retained sampling times are 10.6 hours for NUTS and 0.60 hours for
PMMH, with different GPU workloads and preparation/warmup excluded.

`protocol.json` and `launch.py` describe the original run. A GPU failure
interrupted it after warmup. `resume.py` continues the exact checkpoints
without repeating warmup, then calls `identity_comparison.py`. The final
`nuts-resumed/` checkpoints contain the original warmup prefix followed by
retained draws. `comparison/manifest.json` records the input hashes and
checks of matching targets, starting points, prefixes and frozen tuning.

From the repository root, restore the archived inputs:

```sh
python artifacts/restore_results.py \
  --prefix code/bayesian/corrected/results/identity-adapt-nuts-v1 \
  --prefix code/bayesian/corrected/results/historical-proposal-control-v1
```

Then regenerate the plots and diagnostic table with the reporting dependencies
described in `code/bayesian/corrected/README.md`:

```sh
python code/bayesian/corrected/identity_comparison.py \
  --nuts code/bayesian/corrected/results/identity-adapt-nuts-v1/nuts-resumed \
  --pmmh code/bayesian/corrected/results/historical-proposal-control-v1/pmrw \
  --parent code/bayesian/corrected/results/identity-adapt-nuts-v1/nuts \
  --output code/bayesian/corrected/results/identity-adapt-nuts-v1/comparison \
  --paper
```

The `--paper` option copies three vector PDFs into `imgs/pmcmc` and the
diagnostic table into `results/paper/identity_diagnostics.tex`. All other
generated outputs remain ignored. Completed checkpoints and comparison
outputs are stored in the single `identity-completion-20261019` archive
under `artifacts/packed`; the shared manifest verifies every member.
