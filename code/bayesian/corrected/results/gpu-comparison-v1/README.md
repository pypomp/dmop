# GPU NUTS and PMMH comparison

This is a development run. The primary comparison is now in
[gpu-comparison-v2](../gpu-comparison-v2/README.md), with deterministic GPU
execution and 10,000 likelihood particles. This run's 5,000-particle PMMH
chains completed 30,000 retained draws each under `pmrw-long`, following
500 warmup iterations. That directory includes all draws from `pmrw` and
`pmrw-continued`; do not pool these overlapping files. Its NUTS warmup was
retired and saved. Current NUTS sampling is in
[gpu-forward-nuts-v2](../gpu-forward-nuts-v2/README.md).

The [30,000-draw review](review-pmrw-30000/) contains the frozen draws,
all-parameter traces and marginal densities, trend panels, and diagnostics.
All 18 coordinates were visually inspected. Trend chains largely overlap
(rank split R-hat 1.0075, bulk ESS 542), but seasonal and immigration tails
still differ. Chain 3 has a heavier lower tail for beta3; Chain 1 makes a late
excursion toward omega6 = -20. Across all coordinates, the largest R-hat is
1.0401 (omega4), and the smallest bulk and tail ESS are 132 and 113 (omega6).
This is a particle-count reference, not evidence of joint convergence or
agreement with NUTS. Compare with the 10,000-particle reference when it finishes.

`pmrw-long-completion-verification.json` records process termination,
checkpoint hashes, finite retained states, and frozen step sizes. There were
61 proposals with negative-infinite log acceptance ratios across all four
chains, including warmup. All were rejected with the previous state retained;
these diagnostic entries are not nonfinite sampled states.

From the repository root, reproduce the review in a new output directory:

```sh
python code/bayesian/corrected/report.py \
  code/bayesian/corrected/results/gpu-comparison-v1/pmrw-long/chain{0,1,2,3} \
  --output NEW_REPORT_DIRECTORY
```

`protocol.json` specifies the common target, initialization, prior and checks.
`prepare.py` freezes starts and a proposal covariance from the previous PMMH
development run. The prior is not fitted to these draws. Preparation inputs,
source hashes and its computational cost are recorded in `inputs/manifest.json`.

`launch.py nuts` and `launch.py pmrw` launch the four-chain batches on GPUs 0
and 1 respectively. Both use float64 likelihoods. NUTS uses a float32 proposal
force; PMMH evaluates no gradients. Their checkpoints, configuration and source
snapshots are in `{nuts,pmrw}/chain{0,1,2,3}`. Each chain has an independent seed.
Read the launch records and verify host processes before restarting anything.

The former CPU chains remain development results; do not pool them with these
new chains. The new runs use their iteration-1950 states as starts and a common
regularized covariance estimated from their warmup. Step sizes adapt for 500
iterations; the supplied covariance remains fixed. All adaptation then stops.
Report and inspect the new chains with `../../report.py`. Incomplete or poorly
mixed runs must not replace the manuscript figures.

Both GPUs have unrelated workloads. Per-chain `iteration_seconds` records the
whole batch's wall time. Do not sum it across lanes or interpret the initial
compilation as steady-state sampling cost. Account for shared preparation and
obtain a controlled timing comparison before claiming a computational speedup.
