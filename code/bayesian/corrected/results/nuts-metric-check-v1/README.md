# Proposal covariance check

The depth comparison did not justify longer trajectories. This check asks
whether a better matched proposal covariance improves movement at depth 5.
It changes neither the likelihood nor the proper prior.

The candidate is 75% of the pooled covariance from the unconverged primary
PMMH 5,025-draw snapshot and 25% of the old metric, rescaled to the old trace.
This is proposal tuning, not a fitted prior or a claim that the covariance
estimate is accurate. As an offline check, it reduces the preconditioned
covariance condition number in the independent J5000 snapshot from 34.8 to
9.8. `geometry.json` also shows each chain separately; the snapshots remain
unconverged, and these condition numbers are not a mixing test.

`protocol.json` fixes a paired comparison of the old metric, the candidate
at the same step sizes, and the candidate at twice those step sizes. Each
receives six independent transitions from the same four update-54 states
used in the depth check. Candidate order rotates between trials. They share
one compiled kernel because metric and step size are dynamic arguments.
The separate initial compilation transition is excluded from the score.

After the benchmark finishes, run `summarize.py` to verify the paired seeds
and obtain coordinate-level rate ratios and paired-trial bootstrap intervals.
Each resampled trial keeps all four fixed starting states and its batch
timing. These intervals do not account for uncertainty in the fitted metric,
other posterior states or other hardware workloads.

Consider a candidate only with at least a 50% geometric-mean movement-rate
gain, mean trajectory acceptance at least 0.5, no divergences, and no lower
point rate for beta3 or omega5. Paired uncertainty and all coordinate losses
must also be inspected. This is a local screen, not ESS/sec or convergence.
No production run changes automatically. Any adopted tuning must account
for the PMMH pilot and benchmark costs rather than hiding them in setup.

## Completed comparison

All 18 timed transitions completed with finite states. The covariance-only
candidate had a movement-rate ratio 0.844 (paired interval 0.622--1.278).
The covariance with doubled steps had ratio 3.013 (1.696--6.128), mean
trajectory acceptance 0.622 and zero divergences. Its point rate improved
for all 18 coordinates; beta3 and omega5 ratios were 7.95 and 2.82.
Individual intervals remain broad, and three include a rate loss. The
four fixed states and six trials limit generalization.

The selected candidate passes the prespecified screen and warrants a new
warmup configuration, recorded in `../gpu-forward-nuts-v2`. The benchmark
itself does not establish convergence or production efficiency.
