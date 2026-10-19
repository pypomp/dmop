# Dhaka optimizer A/B comparison

20 paired starts per initialization regime; 1000 fitting particles and 850 seconds per fit. Both methods use the same monthly Gaussian transition and guided smoother. Score updates retain the archived settings, including burn-in 30; numerical SAEM uses burn-in 3 and at most 25 L-BFGS iterations per M-step. This compares the two optimizer packages, including their schedules.

All likelihood axes are linear. Lines in the top panels join estimates from the same start; crosses mark prematurely stopped fits, retaining their last eligible estimate. The bottom panels show SAEM minus score, with twice the paired Monte Carlo standard error. Evaluation draws are shared within each pair and independent of fitting. No fitting output is selected using these evaluations.

Median change subtracts each fit's own initial log-likelihood. Iterations counts completed outer iterations at the selected estimate, including iterations with no parameter change; it is not a count of accepted nonzero steps. Time records the fitting call, including an overrun needed to detect the deadline; over-budget estimates are discarded. Compilation, final evaluation, and the cost of the precomputed IF2 starting points are excluded.
