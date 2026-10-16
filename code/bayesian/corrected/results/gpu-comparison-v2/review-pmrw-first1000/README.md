# First 1,000 retained PMMH draws

This interim report freezes four chains through iteration 1,500, discarding
the prescribed 500 warmup updates. It uses the untempered 10,000-particle
likelihood and common normal prior. The run continues; these are not final
posterior results. Input hashes and the exact retained draws are saved.

All 18 trace and marginal panels, and the four trend panels, were visually
inspected. The axes are linear and the labels, prior overlays and warmup
markers are readable. The traces show substantial serial dependence and
differences between chains, especially in the seasonal coefficients. The
omega6 lower tail is explored unevenly.

Rank R-hat ranges from 1.094 to 1.722; minimum bulk ESS is 6.58. The largest
R-hat is for beta3, with beta2, omega4 and beta1 also poorly mixed. Even the
trend coefficient has R-hat 1.179 and bulk ESS 16.3. Parameter acceptance
over these retained draws is 0.162, 0.285, 0.188 and 0.176.

The larger particle count does not remove slow exploration in this early
sample. This is not a controlled comparison with the longer 5,000-particle
report and does not show that more particles worsen mixing. Continue the
planned run with frozen tuning and review again around 5,000 retained draws
per chain, or sooner if a numerical failure appears. Do not replace the
manuscript figures or claim particle agreement or prior insensitivity from
this report.
