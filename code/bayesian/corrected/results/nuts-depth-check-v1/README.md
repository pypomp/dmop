# NUTS trajectory-depth check

Decision: retain the current depth-5 production run. The completed comparison
does not establish a useful gain from depth 6.

Mean batch time was 67.47 seconds at depth 5 and 134.92 seconds at depth 6.
The geometric-mean movement-rate ratio was 1.214, with paired bootstrap
interval [0.788, 1.626]. This misses the prespecified 25% improvement screen
and leaves substantial uncertainty about any gain. Both depths had zero
divergences and always reached their depth limit; mean trajectory acceptance
was 0.731 versus 0.663. The fifth immigration coefficient, which moved least
in the warmup review, had a rate ratio of 0.960. Its interval is wide, so this
does not establish a loss either. Several other coordinates also had lower
point estimates. Increasing depth would roughly double update cost without
clear evidence of improved efficiency in the difficult coordinates.

`result.json` retains all 14 transitions (two compilation records and twelve
timed records). `paired_summary.json` and `coordinate_comparison.csv` give
the complete summary. The process was verified terminal after completing
all trials. These results concern local movement, not posterior convergence
or ESS per second. No production configuration changed.

The first 50 warmup updates all hit depth 5, with the weakest movement in
seasonal immigration. This benchmark compares depths 5 and 6 at four frozen
warmup states, without changing the production run or its target.

`protocol.json` fixes the states, proposal scales, implementation, paired seeds
and screening rule before launch. Each candidate receives six independent
transitions from each fixed state, with equal random keys across candidates.
Trial order alternates. Compilation is measured separately. Inputs were
captured at update 54; they are distinct from the first-50-update review.

`check.py` verifies the cached likelihood and force, then records all jump
and NUTS diagnostics. The score is the geometric mean of the 18 coordinate
squared-jump rates, scaled by the frozen, unconverged PMMH variances. Timing
is for a whole four-chain batch and must not be added across chains. This is
a local efficiency screen, not an ESS estimate or a posterior comparison.

After completion, `summarize.py` checks the paired seeds and reports each
coordinate's rate ratio. Its bootstrap resamples the six whole paired trials,
retaining all four fixed states and shared batch timings. The intervals
describe variability in this local screen; they do not account for other
posterior states or hardware workloads.

Consider depth 6 only with at least a 25% score improvement, mean trajectory
acceptance at least 0.5, and no divergences. Inspect paired variability and
individual-coordinate losses before any production change. An inconclusive
screen is not grounds for restarting a run. Production remains unchanged.
