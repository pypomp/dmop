# Prior sensitivity

## Review after 60,000 retained draws

The continuation in `pmrw-60000` is complete. `review-pmrw-60000` contains
the frozen draws, diagnostics and plots. `comparison-60000` compares these
four chains with the primary-prior PMMH run's 30,000 retained draws per
chain. All 18 trace and marginal panels, the trend panels, and the prior
comparison were visually inspected. The different lengths are used to
examine the sampled distributions, not to compare computational efficiency.

Doubling the prior standard deviations changes the sampled trend mean
from -0.420 to -0.406 and its standard deviation from 0.112 to 0.115.
The sampled standard deviations of omega1, omega5 and omega6 increase by
factors of 1.81, 1.73 and 2.49. The broad lower tail of omega6 remains
unevenly explored: the four chains spend 9.3%, 10.3%, 25.5% and 0.9% of
their retained draws below -20. Chain 4 now visits this tail, but much less
often than chain 3. Seasonal coefficients also remain strongly dependent
over successive iterations. The largest rank R-hat is 1.074 and the
smallest bulk ESS is 56.6, both for omega6; these have improved from the
30,000-draw review, but do not establish convergence of the joint sample.

The empirical KDE is absent from both settings. These results support its
replacement by an explicit proper prior, while showing that the prior
still affects several immigration coefficients. The size of that effect
remains uncertain because the tails mix slowly. No further sensitivity
continuation is scheduled; the primary NUTS comparison is still running.

`pmrw-60000-completion-verification.json` records finite sampled states,
fixed retained step sizes, and exact preservation of the parent trace and
proposal. Negative infinite log acceptance ratios occur only on rejected
proposals, with the saved state unchanged. The continuation contains the
parent sample; do not pool them as independent draws.

## Review after 30,000 retained draws

The 30,000-draw continuation is complete. `review-pmrw-30000` contains
the frozen report, and `comparison-30000` compares it with the primary
10,000-particle PMMH run at the same length. All 18 trace and marginal
panels, the trend panels, and the prior comparison have been visually
reviewed. Saved states are finite and retained step sizes are fixed;
`pmrw-long-completion-verification.json` records the checkpoint hashes.

The trend mean changes from -0.420 to -0.408 when the prior standard
deviations double, and its sampled standard deviation changes from 0.112
to 0.116. Several immigration coefficients show much larger changes.
The sampled standard deviations of omega1, omega5 and omega6 increase
by factors of 1.98, 1.86 and 2.78. Omega6 has a long lower tail, with
prolonged excursions below -20 in three chains; the fourth chain has
not explored the same tail. Omega1 and omega2 move with these excursions.
Omega5 has a broader lower tail in all four chains.

These results support retaining an explicit prior and reporting its
influence. They do not establish prior insensitivity. The precise changes
are still uncertain because of incomplete mixing: the largest rank R-hat
is 1.147 and the smallest bulk ESS is 21.2, both for omega6. For the trend,
R-hat is 1.020 and bulk ESS is 257.5. The empirical KDE is absent from
both runs; replacing it does not make the choice of prior irrelevant.

`pmrw-60000-protocol.json` specifies a further exact continuation to
60,000 retained draws to examine the unequal tail occupancy. It preserves
the target, random streams and frozen tuning, with no additional warmup.
The continuation includes the previous trace; do not pool the parent and
continuation as separate samples.
The continuation ran in `pmrw-60000` on GPU1. Its launch record
identifies the process, and `pmrw-60000-startup-verification.json` confirms
that the previous 30,500 updates are copied exactly and tuning is unchanged.

This comparison doubles the standard deviations of the common normal prior
in the 18 estimation coordinates. Prior centers and the coordinate
transformation remain fixed. The likelihood is untempered and uses 10,000
particles, as in `../gpu-comparison-v2`. No IF2 parameter cloud defines
either prior.

The four PMMH chains use the primary comparison's starts, fixed covariance,
500 warmup updates and 10,000 retained draws, with fresh seeds. The protocol
and source/input hashes are saved before launch. To avoid adding another
concurrent GPU job, launch only after the existing 5,000-particle PMMH
continuation finishes:

```sh
/home/kevin/anaconda3/envs/pypomp/bin/python code/bayesian/results/prior-sensitivity-gpu-v1/launch.py --check-only
/home/kevin/anaconda3/envs/pypomp/bin/python code/bayesian/results/prior-sensitivity-gpu-v1/launch.py
```

Run these commands from the repository root with host process visibility.
The launcher checks completion, process identity, source and inputs, and
refuses to overwrite an existing run. No background scheduler is installed.

After both settings have converged, compare all parameter marginals, means,
standard deviations and 95% intervals, accounting for Monte Carlo error.
Freeze each with `report.py`, then run `sensitivity.py` with the two report
directories and `--kind prior`. The script verifies that only the prior
scale differs and labels provisional comparisons if either report has not
passed its numerical checks. Its plots retain separate chains.
The long lower tails of omega5 and omega6 in the initial PMMH review deserve
particular attention. Report any material changes. A narrow sampled
distribution relative to one prior does not by itself establish that the
prior has little effect, and one scale comparison cannot establish
insensitivity to every prior choice.

## Review after 10,000 retained draws

All four chains completed. `completion-verification.json` records the terminal
process and checkpoint hashes. `review-pmrw-10000` freezes the complete sample;
all 18 traces and marginal plots, and the trend panels, were visually inspected.
`comparison-10000` compares this sample with the primary J10000 run at the same
retained length. Both runs still show incomplete mixing.

The trend mean is -0.405, versus -0.428 with the original prior; its sampled
standard deviation is 0.118 under both settings. Several immigration
coefficients behave differently. Chain 3 spends long periods below -20 in
omega6 and separates from the other chains in omega1 and omega2. Chain 1
also ends with a downward excursion in omega5. Seasonal transmission
coefficients remain poorly mixed. The largest rank R-hat is 1.406 and the
smallest bulk ESS is 8.54, both for omega6. Acceptance ranges from 0.153 to
0.277. These samples do not establish prior robustness or a converged
estimate of the change caused by widening the prior.

`pmrw-long-protocol.json` specifies an exact continuation to 30,000 retained
draws, preserving the saved states, random streams and frozen tuning. This
continuation includes the original trace as a prefix. Do not combine the two
files as independent samples. The empirical KDE is absent from both settings;
this comparison concerns the width of the replacement normal prior.
