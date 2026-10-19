# NUTS with the tuned proposal covariance

`review-nuts-500` contains the first 500 retained draws from each of the
four chains, after 500 warmup updates. The exact reviewed trace, diagnostics
and step sizes are saved in `review_snapshot.npz`; `numerical_review.json`
records its hash and checks against the report's posterior draws. All saved
states and force norms are finite, and step sizes remain fixed after warmup.
Chain 3 has one divergent trajectory at iteration 821. The sampled state at
that update and subsequent states are finite; no draws have been removed.
Mean trajectory acceptance is 0.612--0.659. Almost every trajectory reaches
the depth limit without a U-turn.

All 18 trace and marginal panels, and the four trend panels, have been
visually reviewed. Chains occupy broadly similar regions, but their marginal
densities still differ, especially for the seasonal and immigration
parameters. The largest rank R-hat is 1.151 (omega3), and the smallest bulk
ESS is 23.4 (beta3). For the trend, R-hat is 1.026 and bulk ESS is 133.9.
This is an interim result; the existing run continues to 2,000 retained
draws per chain without changing its settings.

`comparison-nuts-500` compares this frozen sample with the completed
30,000-draw, 10,000-particle PMMH reference. Both use the same model, prior
and likelihood particle count. The marginal plots show broad overlap across
the 18 coordinates. The trend means are -0.4279 (NUTS) and -0.4203 (PMMH),
with standard deviations 0.1039 and 0.1121. Remaining chain differences and
the single divergent NUTS trajectory prevent a convergence claim. This
unequal-length comparison does not establish a runtime advantage.

To regenerate the method comparison from the frozen reports, run from the
repository root with the reporting dependencies installed:

```sh
python code/bayesian/compare.py \
  --nuts code/bayesian/results/gpu-forward-nuts-v2/review-nuts-500 \
  --pmmh code/bayesian/results/gpu-comparison-v2/review-pmrw-30000 \
  --output /tmp/dmop-nuts-pmmh-comparison
```

This is the primary NUTS comparison following the completed covariance screen
in `../nuts-metric-check-v1`. The untempered Pypomp likelihood, common proper
prior, particle counts, force, and four starting points are unchanged.
The proposal covariance and initial step size change. Each chain uses a fresh
seed and 500 warmup updates before 2,000 retained draws. Earlier warmup is
preserved separately and is never pooled as posterior draws.

The selected screen candidate improved the local geometric-mean movement rate
by 3.01 times (paired bootstrap interval 1.70--6.13), with mean trajectory
acceptance 0.622 and no divergences. All coordinate point estimates improved.
This is evidence for tuning, not posterior convergence or an ESS/sec speedup.
The common initial step 0.07 approximates the tested lane-specific steps;
the existing warmup rule adapts each lane separately.

Run `launch.py --check-only`, then `launch.py`, from the repository root with
host GPU/process access. Keep the earlier NUTS process until a finite first
checkpoint and source/target verification; then record its retirement. Both
PMMH runs continue unchanged. Review movement after 50 warmup updates.

Any cost comparison must include original initialization, the PMMH covariance
pilot, retired NUTS warmup, both tuning screens, and new warmup. No final
convergence, prior-insensitivity or speedup claim is supported yet.

Use the same 50-update review as the preceding configuration:

```sh
PYTHONPATH=/tmp/dmop-bayes-deps MPLCONFIGDIR=/tmp/dmop-matplotlib \
/home/kevin/anaconda3/envs/pypomp/bin/python \
  code/bayesian/results/gpu-forward-nuts-v1/review_warmup.py \
  --run-root code/bayesian/results/gpu-forward-nuts-v2
```

It refuses to run before all four chains have 50 saved warmup updates, or to
overwrite an existing review. The PMMH reference intervals remain the same
frozen, unconverged snapshot; they are descriptive scales, not known posterior
intervals. The output is `warmup-review-50` under this directory.

`warmup-health-250` freezes the first 250 traces, diagnostics and step sizes
from each chain. All states and force norms are finite, with no numerical
divergences. Over iterations 201--250, mean trajectory acceptance is
0.591--0.644 and particle-randomness acceptance is 0.66--0.74. All four chains
move at every update in that window. Trajectories reach the depth limit
without a U-turn in 98--100% of updates, so this check does not establish
efficient exploration. Settings remain unchanged. These are adaptive warmup
draws, not retained posterior samples; the next posterior review is after
500 retained draws following the 500-update warmup.

`warmup-exit-502` freezes the first 500 warmup updates and two retained
updates per chain. `verify_warmup_exit.py` checks that step sizes remain
constant after warmup, the covariance matches its supplied value exactly,
and saved states and current forces are finite. These checks pass for all
four chains, with no numerical divergences in the reviewed updates. All
trajectories in the last 50 warmup updates reached the depth limit. This
is a numerical and adaptation check, not evidence of posterior convergence
or efficient mixing. The next posterior review remains at 500 retained draws.
