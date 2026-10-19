# Primary J10000 PMMH review

This snapshot contains 5,025 retained draws per chain after 500 warmup
updates. `audit.json` identifies the exact checkpoints and frozen draw file.
All 18 trace and marginal panels and the four trend panels were visually
inspected.

The chains have not converged. Rank split R-hat ranges from 1.019 to 1.179;
minimum bulk ESS is 18.6 and minimum tail ESS is 20.0. The worst disagreement
is now in the second and third seasonal transmission coefficients, followed
by the first and fifth. The third chain makes long excursions in several
immigration coefficients, including a lower-tail excursion in omega2, so
the remaining difficulty is not confined to omega5 and omega6.

The trend marginal curves overlap reasonably, but their traces remain
autocorrelated. The second chain still has autocorrelation around 0.46 at
lag 100. Thus the four trend panels alone would give an overly reassuring
picture. Acceptance rates are 0.198, 0.238, 0.212 and 0.178.

Agreement has improved since the 1,000-draw review, whose maximum R-hat was
1.722, but these results cannot yet support posterior comparisons or claims
of prior insensitivity. Differences from the separately seeded J5000 run
are not evidence of a particle-count effect while either run remains
unconverged.

Continue the existing planned run to 10,000 retained draws, then review all
coordinates again and extend or retune if needed. NUTS is still warming up;
the deeper-trajectory screen did not justify changing its current setting.
These are development figures, not replacement manuscript results.
