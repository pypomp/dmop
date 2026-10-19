# Deterministic GPU comparison

The PMMH run here remains part of the primary comparison, continued in
`pmrw-long` with its original tuning and saved draws preserved. It has completed
30,000 retained draws per chain; see the review below. NUTS now runs in
[`../gpu-forward-nuts-v2`](../gpu-forward-nuts-v2), using a validated forward
derivative of the same proposal objective and a revised proposal covariance.
The first forward-mode run is retained in `../gpu-forward-nuts-v1` as a
retired warmup. The reverse-mode NUTS warmup here
was retired at 29 updates after the new run passed its startup checks; see
`nuts/termination.json`. Preserve its cost and checkpoints.

Both methods use 10,000 particles
for the untempered likelihood and the same common proper prior. NUTS uses
1,000 particles for its float32 proposal force; likelihoods remain float64.
The two deterministic XLA flags are recorded in `protocol.json` and each run.

The [shared inputs](../gpu-comparison-v1/inputs/manifest.json) specify four
starting points and the PMMH proposal covariance. The covariance was estimated
from earlier pilot PMMH draws and stays fixed; its overall scale adapts for
500 iterations and then stays fixed. Current NUTS uses the same starts with
the revised covariance documented in its run directory. Preparation costs
must be included in computational comparisons.

The [expanded noise check](../posterior-noise-check-v1/expanded-summary.json)
motivates the higher particle count. The [GPU replay check](../batch-continuation-check-v1/optimized-verification.json)
verifies bitwise continuation of four short real-model chains with these XLA
flags, including interrupted warmup and unequal saved iteration counts.

`launch.py nuts` and `launch.py pmrw` start the batches on GPUs 0 and 1.
Their configurations, frozen source and checkpoints are under
`{nuts,pmrw}/chain{0,1,2,3}`. Both GPUs share unrelated work. Read launch records
and verify host processes before restarting anything. Use `../../batch_extend.py`
for continuation, with the recorded XLA flags and original batch shape.

The previous 5,000-particle PMMH run is retained for a particle-count check.
It is not pooled with these chains. The previous NUTS warmup used default
GPU execution, whose float32 forces were not bitwise repeatable; it remains
development evidence. No run replaces manuscript figures until convergence
and sensitivity checks are satisfactory.

## Review of the completed PMMH run

[`review-pmrw-2000`](review-pmrw-2000) contains the first 2,000 retained
draws per chain, matching the planned NUTS length. All parameter plots were
inspected and the saved prefix was verified exactly. This supports a
comparison at equal iteration counts, not equal runtime. The longer report
below remains the reference for checking the sampled distributions.

`review-pmrw-30000` contains the frozen draws, all-parameter traces and
marginals, trend panels, and diagnostic table. Each of the four chains has
500 warmup iterations followed by 30,000 retained draws. The continuation
includes the original run's draws; do not pool `pmrw` with `pmrw-long`.
`pmrw-long-completion-verification.json` records checkpoint hashes and checks
of finite sampled states and fixed retained-sampling step sizes. The 48
negative-infinite log acceptance ratios occur on rejected proposals, with
the sampled state unchanged.

All 18 traces and marginals and the trend panels were visually inspected.
The chains overlap much better than at 10,000 retained draws. For
100 times the trend coefficient, the mean is -0.4203, standard deviation
0.1121, and central 95% interval [-0.6514, -0.2098]. Its rank R-hat is 1.0035
and bulk ESS is 536. Autocorrelation remains substantial at lag 100.
Seasonal coefficients and immigration parameters still have unequal tail
occupancy, especially omega6. Across all parameters, the largest R-hat is
1.0135 (omega5), smallest bulk ESS is 202 (beta3), and smallest tail ESS is
152 (omega6). These results do not establish joint convergence.

`particle-comparison-30000` compares these draws with the completed
5,000-particle reference at the same retained length. All 18 comparison
panels were visually inspected. The marginal distributions largely overlap;
the largest absolute mean shift is 0.176 of the corresponding
10,000-particle marginal standard deviation. The trend means differ by
0.0030. The 5,000-particle omega5 marginal has an 18% larger standard
deviation, and some seasonal and immigration tails remain unevenly sampled.
Neither run passes every numerical convergence check, so this comparison
does not establish particle-count insensitivity. Both are retained as
posterior references while NUTS and the wider-prior run finish. No additional
PMMH continuation was launched at this review.

From the repository root, regenerate into fresh output directories:

```bash
python code/bayesian/corrected/report.py \
  code/bayesian/corrected/results/gpu-comparison-v2/pmrw-long/chain{0,1,2,3} \
  --output NEW_REPORT_DIRECTORY
python code/bayesian/corrected/sensitivity.py \
  --reference NEW_REPORT_DIRECTORY \
  --alternative code/bayesian/corrected/results/gpu-comparison-v1/review-pmrw-30000 \
  --kind particles --output NEW_COMPARISON_DIRECTORY
```
