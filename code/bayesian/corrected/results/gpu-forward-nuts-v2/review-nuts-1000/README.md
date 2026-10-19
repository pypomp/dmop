# NUTS review after 1,000 retained draws per chain

This review uses exactly iterations 501--1500 of each of the four chains.
The live run continues to its original endpoint of 2,000 retained draws per
chain. No samples were removed and no sampler settings were changed.

The trend panels, all 18 parameter traces and marginal densities, and the
NUTS/PMMH marginal comparison were inspected. The central distributions
overlap across chains. Immigration tails remain uneven: chain 3 has a
larger upper tail for omega1 and omega3, a larger lower tail for omega2,
and a long low excursion for omega6. Its epsilon trace also spends time
above the other chains near iterations 1100--1250. The seasonal parameters
have overlapping distributions but persistent dependence in the traces.

The largest rank R-hat is 1.088 (omega1), compared with 1.151 at the
500-draw review. The smallest bulk ESS is 37.0 (omega1); the smallest tail
ESS is 90.2 (epsilon). These are improvements in overall chain agreement,
but do not establish joint convergence. Trend R-hat is 1.016, bulk ESS
216.9 and tail ESS 423.9. Trend autocorrelation falls quickly at short
lags, with a smaller positive rebound in chains 1 and 4 around lag 50.

Two trajectories were flagged as divergent, both in chain 3, at iterations
821 and 1206. The first was already present in the 500-draw review. Saved
states and force norms are finite, including at and after these events.
The retained step sizes are unchanged. All trajectories reached depth 5;
the divergent trajectories used 29 and 30 integration steps instead of 31.
Finite saved states do not rule out sampling error from the trajectory
construction. The full run will be reviewed before deciding on further
changes.

The comparison in `../comparison-nuts-1000` uses the completed 30,000-draw
PMMH reference with the same model, prior and 10,000 likelihood particles.
All 18 marginal distributions broadly overlap. For the displayed trend
coordinate, the NUTS mean is -0.4410 (SD 0.1058), versus -0.4203
(SD 0.1121) for PMMH. The difference is -2.39 times the combined estimated
Monte Carlo standard error. The largest such difference is 2.62 for log
sigma. These estimates remain subject to incomplete mixing; the figures
do not establish equivalence or a runtime advantage.

`review_snapshot.npz` preserves the exact trace, diagnostic, step-size and
timing arrays used here, including warmup. Its first 1,000 iterations match
the earlier review bit for bit. `numerical_review.json` records these checks.
The report was generated with:

```bash
env PYTHONPATH=/tmp/dmop-bayes-deps MPLCONFIGDIR=/tmp/dmop-matplotlib \
  /home/kevin/anaconda3/envs/pypomp/bin/python code/bayesian/corrected/report.py \
  code/bayesian/corrected/results/gpu-forward-nuts-v2/nuts/chain{0,1,2,3} \
  --max-retained 1000 \
  --output code/bayesian/corrected/results/gpu-forward-nuts-v2/review-nuts-1000
```

Run this from the dmop repository root. The environment paths describe the
local execution; Python dependencies are listed in the parent code README.
