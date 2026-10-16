# Deterministic GPU comparison

This is the primary NUTS and PMMH comparison. Both methods use 10,000 particles
for the untempered likelihood and the same common proper prior. NUTS uses
1,000 particles for its float32 proposal force; likelihoods remain float64.
The two deterministic XLA flags are recorded in `protocol.json` and each run.

The [shared inputs](../gpu-comparison-v1/inputs/manifest.json) are unchanged
from the previous comparison. They specify four starting points and a common
proposal covariance. The covariance stays fixed; step sizes adapt for 500
iterations. Preparation costs must be included in computational comparisons.

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
