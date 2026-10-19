# First 50 warmup updates with the tuned covariance

All 18 movement panels were visually reviewed. There are no divergences;
mean trajectory acceptance is 0.595--0.643 across chains. All trajectories
reach depth 5 (31 integration steps). The median batch update takes 40.41 seconds
on the observed shared hardware. Neither the depth limit nor these timings
establish posterior efficiency.

Several previously slow seasonal directions move more than in the earlier
50-update review. RMS jumps in beta2, beta3 and omega5 are respectively
1.44, 1.44 and 1.47 times the earlier values. `movement_comparison.csv` gives
all 18 coordinates, combining the four per-chain squared RMS values before
taking a square root. The runs use different seeds and visit different
states while adapting, so these ratios are descriptive, not a paired
comparison or confirmation of the screen's threefold movement-rate gain.

The chains remain visibly separated in gamma, beta1/beta2/beta3 and several
immigration coefficients. Lower immigration tails are still incompletely
explored. Grey intervals come from the same frozen, unconverged J5000 PMMH
snapshot used in the previous review; they are not known posterior intervals.

Keep the current configuration. Let its 500-update warmup finish, then review
all 18 posterior diagnostics after 500 retained draws. No warmup samples enter
the posterior report. The planned 2,000 retained draws may require extension;
no convergence, method-agreement or prior-insensitivity claim is made.
