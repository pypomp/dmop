# Wider-prior PMMH: 60,000 retained draws

Four chains, each with 500 warmup updates and 60,000 retained draws, from
`../pmrw-60000`. The likelihood is untempered, uses 10,000 particles, and
the prior standard deviations are twice those of the primary comparison.
The first 30,000 retained draws match `../review-pmrw-30000` exactly.

All 18 traces and marginals and the four trend panels were visually
inspected. Core distributions overlap, but seasonal coefficients remain
strongly dependent and immigration tails have unequal occupancy. Chain 3
has the greatest upper omega1 and lower omega2/omega6 tail occupancy.
Chain 4 now visits the lower omega6 tail, but spends much less time there.
The trend autocorrelation is still about 0.35 at lag 100.

The largest rank R-hat is 1.074 and the smallest bulk ESS is 56.6, both
for omega6. For the trend, the mean is -0.4056, SD is 0.1151, rank R-hat
is 1.0146, bulk ESS is 503.2, and mean MCSE is 0.0051. These are descriptive
summaries; the joint sample has not met the numerical convergence checks.

`../comparison-60000` gives the comparison with the primary-prior PMMH
sample. Widening the prior has little effect on the sampled trend width,
but materially broadens several immigration distributions. See the parent
README for the conclusion and limits. This sensitivity run is stopped at
its planned endpoint, with no further continuation scheduled.

Reproduce from the repository root:

```sh
python code/bayesian/report.py \
  code/bayesian/results/prior-sensitivity-gpu-v1/pmrw-60000/chain{0,1,2,3} \
  --output /tmp/dmop-wide-60000-review
python code/bayesian/sensitivity.py \
  --reference code/bayesian/results/gpu-comparison-v2/review-pmrw-30000 \
  --alternative /tmp/dmop-wide-60000-review --kind prior \
  --output /tmp/dmop-wide-60000-comparison
```

Use the Python dependencies documented in the corrected experiment's
README. The output paths above must be fresh for the comparison command.
