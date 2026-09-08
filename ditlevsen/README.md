# Ditlevsen transition-density benchmark on Dacca

This directory tests Reviewer 1's proposed hypoelliptic transition-density
competitor as a computational-statistics method. The final 100-start Dacca
experiment has **not** been launched. It is gated on small validation runs.

## What is implemented

1. A paper-faithful implementation of Ditlevsen and Samson's conditional
   proposal particle filter for their harmonic oscillator, checked against the
   analytic linear-Gaussian filter.
2. The DS second-order mean and first-bracket Gaussian covariance for the
   active Dacca coordinates, with an explicitly reported numerical nugget.
3. A bootstrap path-space SMC for Dacca's noisy monthly death observations.
   At every stochastic-approximation iteration it imputes a new complete path
   and evaluates its complete Gaussian pseudo-likelihood and score. Dacca is
   not a demonstrated curved exponential-family example, so there is no SAEM
   M-step: the implementation performs projected Gaussian pseudo-score ascent
   instead. This is not the score of the original Dacca process.
4. A fixed-lower/fixed-upper blocked-interior conditional SMC kernel derived
   from the bridge construction in Karppinen--Singh--Vihola (KSV) Algorithm 8.
   It is explicitly not labeled full CPF-BBS because it cannot reconnect the
   upper endpoint to another lower-boundary forward particle.

The former EKF/QML optimizer is not used as the DS result. Its outputs have
been moved to `results/withdrawn_qml/` because they were generated before SMC
was implemented and included a published-point start. They are retained only
as recoverable diagnostics.

## Why the Dacca method is an extension

For the active state

`(S, I, R1, R2, R3, monthly_deaths)`

Dacca has one Brownian driver and rank-one instantaneous diffusion away from
`S I = 0` (rank zero on that boundary). That is the intended hypoelliptic
structure. Ditlevsen--Samson's model has one
smooth coordinate and a directly forced rough block of dimension `d-1`; its
first drift bracket makes the leading order-1.5 covariance nondegenerate. At
the manuscript state, Dacca's numerical rank diagnostic fills the state only
through bracket depth five; this is not a proof of global hypoellipticity. Its
literal first-bracket covariance is rank at most two in six dimensions. The
runnable SMC adds a disclosed nugget;
the optional six-direction controllability expansion is a new approximation,
not a claim made in the DS paper.

Dacca also observes no state coordinate exactly. It observes a noisy monthly
death count. Therefore the Dacca SMC uses a bootstrap proposal rather than the
paper's conditional proposal `p(U_i | V_i, V_{i-1}, U_{i-1})`.

## Validation status

The Dacca process covariates are evaluated at Euler left endpoints and the
population used by the measurement model is evaluated at the observation
time, matching Pypomp. This fixed a previous one-substep indexing error.

The literal harmonic-oscillator checkpoint uses the paper's values
`D=4`, `gamma=0.5`, `sigma=0.5`, `Delta=0.02`, 100 particles, and 1,000 points.
For the recorded validation trajectory, particle and analytic filtered means
differ by RMSE 0.00523; their RMSEs against the simulated hidden coordinate are
0.03881 and 0.03852. Median ESS is 76.6/100.

Full-series Dacca filter checks from three genuine bounding-box draws give:

- one and two substeps per month: complete invalid-particle collapse;
- five, ten, and twenty substeps: finite likelihoods and surviving particles;
- all viable path-space runs: one surviving early ancestor after 600 monthly
  resampling operations.

On a known-truth oscillator trajectory, the numerical SMC-score extension
starts from `(D, gamma, sigma)=(3, 1, 0.8)` and reaches approximately
`(3.36, 0.54, 0.50)` from truth `(4, 0.5, 0.5)`. Its observed likelihood is
within 0.54 log units of the trajectory-specific numerical MLE. This validates
the score construction used for Dacca; it is not a reproduction of DS's exact
sufficient-statistic M-step or their 100-dataset Table 1.

The corrected ten-update Dacca pilot increased both the independently estimated
DS-SMC likelihood and the common Euler-20 target for three bounding-box starts.
Those changes were only tens of log units at the original learning rate 0.001,
which the preflight showed was far too small for the manuscript's wide box.

A 20-update preflight at learning rate 0.05 improved fresh Euler-20 estimates by
1,521, 1,740, and 4,606 log units for the first three box draws. Two runs
completed; the first stopped after 19 updates when backtracking could not keep
the maximum invalid-particle fraction below 50%. A separate aggressive
60-update stress test improved Euler-20 from about -6209 to -4015 in 82 seconds,
but was non-monotone and drove `tau` to its upper bound. These checks establish
a real SMC fitting signal, but also show that the current optimizer is not yet
robust enough for the unattended final 100-start run. The exact preflight
summaries and qualifications are in `results/validation/PREFLIGHT.md`.

Every viable full-series forward genealogy has only one surviving early
ancestor. The fixed-endpoint bridge changes within-month interiors but cannot
change that genealogy and had mixed effects on short-fit progress. KSV is not
needed for the DS Algorithm 2 baseline, which reruns an independent forward SMC
at each stochastic-approximation iteration. The blocked kernel is therefore
excluded from the planned headline fit; a true KSV comparison would require a
full forward CPF-BBS implementation and a separate invariance/mixing check.

## Reproduction

From this directory:

```bash
PYTHONPATH=.:../.. \
  /home/kevin/anaconda3/envs/pypomp/bin/python -m pytest -q tests

PYTHONPATH=.:../.. \
  /home/kevin/anaconda3/envs/pypomp/bin/python -m ditlevsen.validate

PYTHONPATH=.:../.. \
  /home/kevin/anaconda3/envs/pypomp/bin/python -m ditlevsen.validation_plots
```

All final searches will be generated by `make_starts`: every start is an
independent reproducible draw from the same physical-scale box used in
`../code/global_search/prep.py`. There is no published-point exception. Every
candidate will be evaluated under the same 20-substep Euler POMP used by the
DMOP manuscript. Final likelihood evaluation will use 5,000 particles and 36
replications, matching the manuscript; the current lower-particle runs are
validation only.

## Layout

- `ditlevsen/validation_ho.py`: literal DS oscillator filter and analytic oracle.
- `ditlevsen/smc.py`: Dacca filter, path imputation, and score update.
- `ditlevsen/smc_benchmark.py`: resumable global-box benchmark and independent
  Euler-20 likelihood evaluation.
- `ditlevsen/kbridge.py`: optional fixed-endpoint blocked-interior bridge kernel.
- `ditlevsen/transition.py`: local hypoelliptic Gaussian approximations.
- `ditlevsen/bridge.py`: exact linear-Gaussian block and bridge conditionals.
- `method.tex`: assumptions, rank analysis, and combined-algorithm derivation.
- `invertibility_note.tex` and `invertibility_note.pdf`: standalone derivation of
  why the published DS covariance is invertible but the literal Dacca analogue
  is singular.
- `literature_notes.md`: source audit of DS, KSV, and the reviewer claim.
- `results/reference/`: exact IFAD/IF2 manuscript results exported from the
  original result objects.
- `results/validation/`: current numerical preflight results; these are not the
  final 100-start experiment. The PNGs show the likelihood comparison,
  manuscript-style parameter panel, elapsed-time trace, and substep viability.
- `results/withdrawn_qml/`: rejected EKF/QML diagnostics, not final results.

Nothing here commits, pushes, or changes the Pypomp package.
