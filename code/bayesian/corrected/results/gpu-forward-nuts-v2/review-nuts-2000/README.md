# Completed primary NUTS review

Four chains, 500 warmup and 2,000 retained iterations each; all draws retained.
`numerical_review.json` verifies the completed checkpoints against the frozen
snapshot, exact reconstruction of displayed coordinates, the earlier
1,000-draw review as an unchanged prefix, finite saved states/forces and
fixed retained step sizes. `audit.json` preserves input hashes and diagnostics.

Trend R-hat is 1.014, bulk ESS 333, tail ESS 745, and mean MCSE .00593.
Across all 18 coordinates, maximum R-hat is 1.059 and minimum bulk ESS is
60 (omega1); minimum tail ESS is 116 (omega6). Two divergences occur in
chain 3 at total iterations 821 and 1206. Almost all trajectories reach the
depth-5 cap. These diagnostics do not establish joint convergence.

All 18 trace and density panels were inspected. Central distributions overlap,
but omega1 and omega6 retain long excursions and unequal tail exploration.
The trend traces have no sustained drift. Comparison with the longer PMMH
reference shows broad marginal agreement, with residual differences in tails.
There is no computing-time or prior-insensitivity claim.

Reproduce the report from the dmop root:

```sh
python code/bayesian/corrected/report.py \
  code/bayesian/corrected/results/gpu-forward-nuts-v2/nuts/chain{0,1,2,3} \
  --output NEW_REPORT_DIRECTORY
```

Use `paper_figures.py` for the manuscript layouts, paired table and manifest.
