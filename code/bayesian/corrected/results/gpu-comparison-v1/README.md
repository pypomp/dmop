# GPU NUTS and PMMH comparison

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
