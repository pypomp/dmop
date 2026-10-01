# Monthly order-1.5 block pilot

This is the requested ten-start check before any 100-start experiment. Each
fit began at an independent draw from the manuscript's global-search box and
ran for at most 800 seconds. No published parameter estimate was used as an
initial value.

At each of 20 internal steps per month, the transition uses the DS
strong-order-1.5 mean and the moment covariance of the scalar-noise stochastic
Taylor step. This includes the multiplicative-noise terms involving `L0 g`,
`L1 g`, and `L1 L1 g`. The 20 local Gaussians are propagated through a
time-varying linearization and the 19 intermediate states are integrated out.
SMC samples and scores only the resulting monthly endpoints. A `1e-12`
relative floor is applied once to the monthly covariance; there is no local
nugget.

The optimizer uses 100 particles per update and a one-path Fisher-identity
pseudo-score. It is a Robbins--Monro score method for the block-Gaussian
pseudo-model, not the closed-form SAEM M-step in DS Algorithm 2. Final
parameters were evaluated under the manuscript's Euler-20 model with 5,000
particles and 36 replications.

## Results

| Statistic | Euler-20 log likelihood |
|---|---:|
| Maximum | -3873.47 |
| Median | -4440.63 |
| 10th percentile | -5108.83 |
| Fits at or above -3780 | 0/10 |
| Comparable IFAD-0.97 maximum | -3744.17 |
| Comparable IFAD-0.97 median | -3745.77 |

The ten fits completed between 197 and 300 score updates. Several improved
quickly and then deteriorated. At the 100-second trace checkpoints, the median
Euler-20 estimate was -4388.47. It was -4049.73 at 500 seconds and -4442.17 at
800 seconds. The best low-effort trace checkpoint was -3860.67 at about 202
seconds, but this was evaluated with only 1,000 particles and four
replications and is not a final result.

Seven final fits put `tau` at its upper bound of 0.5, and three put `sigma` at
its upper bound of 5. The block likelihood improved in nine fits, but that
surrogate improvement did not reliably track the Euler-20 target late in the
run.

The block construction fixes the local-density problem well enough to run a
stable monthly particle filter, and the early optimization signal is real. It
does not outperform IFAD under the 800-second budget. The 100-start experiment
should not be launched without changing the optimization rule, for example by
retaining the best independently evaluated checkpoint or reducing the late
learning rate.

The manuscript-cutoff figures omit likelihoods below -3780, so the Ditlevsen
row is empty. The files with `full` in their names show all ten Ditlevsen fits.
All figures are PNG files.
