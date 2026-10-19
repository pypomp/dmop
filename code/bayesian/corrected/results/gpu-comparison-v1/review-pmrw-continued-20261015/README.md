# Interim PMMH review

This freezes 2,338 retained draws per chain from the four continued
5,000-particle PMMH chains, through iteration 2,838. It is a development
report, not a replacement manuscript result. The chains continue running.
`audit.json` records the input hashes and configuration; `posterior_draws.npz`
contains the exact retained draws used in the diagnostics and marginal plots.

All 18 trace and marginal panels, and the four trend panels, were visually
inspected. Labels, prior overlays and warmup markers are readable, with
linear axes. The chains overlap but retain substantial serial dependence.
Rank R-hat ranges from 1.033 to 1.235; the minimum bulk ESS is 12.7.
The trend coefficient has R-hat 1.033 and bulk ESS 78.9, so its comparatively
good appearance does not establish convergence of the whole parameter vector.

The seasonal immigration coefficients move slowly. In particular, omega5
and omega6 have long lower tails, and their sampled distributions are less
concentrated relative to their priors than most other coordinates. This
identifies parameters to scrutinize in the planned prior-scale check; it
does not establish prior sensitivity from this single, unconverged run.

Continue the existing sampler with its frozen tuning. Review the matched
10,000-particle run after it has accumulated retained draws, then assess
particle agreement and prior sensitivity. Do not combine the immutable
parent and its continuation as independent chains.
