# PMMH after 10,000 retained draws per chain

This report freezes the completed four-chain J5000 continuation. The report
contains the parent prefix once, not a pool of overlapping runs. All 18
trace and marginal panels and the four trend panels were visually reviewed.

Agreement improves, but convergence is not established. Rank R-hat ranges
from 1.013 to 1.087; minimum bulk ESS is 42.5 and minimum tail ESS is 57.5.
Omega4 has the largest R-hat, omega6 the smallest bulk ESS, and beta3 the
smallest tail ESS. Chain3 has a pronounced late excursion in beta1/beta2/
beta3 and the trend; omega5 and omega6 lower tails remain unevenly explored.
Trend marginals broadly overlap, but autocorrelation remains substantial.

`../pmrw-long-protocol.json` specifies 20,000 further updates per chain,
retaining the exact saved state and tuning, for 30,000 retained draws total.
The wider-prior J10000 study runs separately. Neither this report nor a
comparison to an unconverged J10000 report establishes particle agreement
or prior insensitivity. These figures have not replaced the manuscript.
