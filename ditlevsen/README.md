# Ditlevsen transition-density benchmark on Dacca

This directory tests Reviewer 1's proposed hypoelliptic transition-density
competitor as a computational-statistics method. The corrected 100-start
Dacca fits and final likelihood evaluations are complete in
`results/block_smc_guided_j5000_final_100/`; the manuscript-style PNGs are in
its `figures/` subdirectory. Its 5,000-particle settings were
chosen using `results/particle_sweep_cached/` and
`results/j5000_focused_tuning/`. The 50 completed fits from the earlier
100-particle attempt are retained as a diagnostic in
`results/block_smc_guided_100/`. Every search starts from an independent draw
from the manuscript box. The local-transition and monthly bootstrap pilots are
diagnostics only.

## What is implemented

1. A paper-faithful implementation of Ditlevsen and Samson's conditional
   proposal particle filter for their harmonic oscillator, checked against the
   analytic linear-Gaussian filter.
2. The DS second-order mean and the full moment covariance of the scalar-noise
   strong-order-1.5 Taylor step. The covariance includes the multiplicative-
   noise terms involving `L0 g`, `L1 g`, and `L1 L1 g`; it reduces to DS
   equation (34) for the additive-noise harmonic oscillator.
3. A 20-step monthly Gaussian block transition. The code propagates the DS
   mean and covariance through the internal steps, integrates out the 19
   intermediate states under a time-varying linearization, and adds one
   `1e-12` relative covariance floor at the monthly endpoint. No floor is
   added to the internal transitions.
4. A guided path-space SMC for Dacca's noisy monthly death observations.
   Particles are proposed and scored only at monthly endpoints. The Gaussian
   proposal freezes the heteroskedastic observation variance at the predicted
   death count and conditions the block transition on that linear-Gaussian
   surrogate. The particle weight includes the exact block-model `p/q`
   correction and the original measurement density. At
   every stochastic-approximation iteration, FFBSi imputes a monthly path and
   the method evaluates its complete block-Gaussian pseudo-likelihood and
   score. Dacca is not a demonstrated curved exponential-family example, so
   there is no SAEM M-step: the implementation performs projected pseudo-score
   ascent instead. This is not the score of the original Dacca process.
5. A fixed-lower/fixed-upper blocked-interior conditional SMC kernel derived
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
local order-1.5 covariance is rank at most two in six dimensions. The monthly
block covariance can nevertheless become full dimensional because the local
covariances are transported through different drift linearizations and summed.
It remains extremely ill-conditioned, so the runnable block SMC adds one
disclosed floor to the endpoint covariance. The optional six-direction
controllability expansion is a separate diagnostic, not a claim made in the DS
paper.

Dacca also observes no state coordinate exactly. It observes a noisy monthly
death count. The Dacca SMC therefore uses an observation-guided Gaussian
proposal formed by freezing the observation variance at the block-predicted
death count. The importance weight uses the original measurement density and
the exact block-model `p/q` correction. This is analogous in purpose to the
paper's conditional proposal `p(U_i | V_i, V_{i-1}, U_{i-1})`, but it is not
the same proposal because the observation structures differ.

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
but was non-monotone and drove `tau` to its upper bound. These checks established
a real SMC fitting signal and motivated the guided proposal, FFBSi smoother,
learning-rate decay, and periodic likelihood guard used in the current run.
The exact preflight summaries and qualifications are in
`results/validation/PREFLIGHT.md`.

The superseded local-transition ten-start pilot allowed up to 1,000 seconds for each fit and
used the manuscript's final evaluation effort of 5,000 Euler-20 particles and
36 replications. For 5, 10, and 20 inference substeps per month, respectively,
the Euler-20 medians were -3847.94, -3821.47, and -3846.53; maxima were
-3786.13, -3776.58, and -3761.41. Ten and twenty substeps each produced one
fit above -3780; five produced none. The complete summary and PNG figures are
in `results/smc_benchmark_10/`. Those numbers must not be presented as results
from the monthly block method.

The superseded monthly bootstrap pilot used ten bounding-box starts, 100
particles per score update, and an 800-second limit. Final estimates were evaluated independently
with 5,000 Euler-20 particles and 36 replications. The best, median, and 10th
percentile log likelihoods were -3873.47, -4440.63, and -5108.83; no fit
exceeded -3780. Comparable-effort IFAD-0.97 has best and median values -3744.17
and -3745.77. The block method therefore remains clearly worse under this
budget. Seven fits ended with `tau=0.5`, its upper bound, and three with
`sigma=5`, also its upper bound. The elapsed-time checks show that several runs
improve early and deteriorate later: the pilot median is about -4050 at 500
seconds and -4442 at 800 seconds. Results and PNG figures are in
`results/block_smc_10/`. These results predate the guided proposal, FFBSi
smoother, and periodic likelihood guard, and are not the headline comparison.

The guided-FFBSi attempt in `results/block_smc_guided_100/` was stopped after
50 of its planned 100 starts when the fitting particle count was increased
from 100 to 5,000. Independent Euler-20 evaluation with 5,000 particles and 36
replicates gave a median log likelihood of -4020.64 and a best value of
-3819.36; none of the 50 fits exceeded -3800. The matching Figure 3, Figure 4,
and elapsed-time PNGs are in its `figures/` directory. The elapsed-time curve
uses separate Euler-20 evaluations with 5,000 particles at every recorded
optimizer step.

The initial 5,000-particle implementation required about 33.4 seconds per
ordinary score update. Replacing materialized Hessians by equivalent
directional derivatives and keeping the forward particle cloud on the GPU
through the backward pass first reduced this to about 19.6 seconds on the same
hardware. The filter was also recomputing every 20-step transition during
backward sampling. Retaining those Gaussian means and covariances from the
forward pass reduced a warmed-up smoothing-path draw from 17.0 to 10.7 seconds;
a full warmed-up path-and-score update took 15.6 seconds in a separate timing
run. The revised and former paths agree bit for bit at fixed seeds in the
short validation cases. For comparison, one IFAD Adam iteration in the
manuscript timing data takes about 3.5 seconds. The remaining difference is
structural: a DS update propagates a 6-dimensional mean and 6-by-6 covariance
through 20 internal steps for every particle and evaluates all backward
transition densities. IFAD performs one Euler simulation/filter pass and has
no backward smoother. As in SAEM-SMC, the stochastic-approximation update uses
one sampled smoothing path. Raising the particle count improves the
approximation to that path's smoothing distribution, but it does not average
5,000 independent complete-path scores. This is why the extra particles can
cost much more than they improve learning per second.

The 400-second tuning sweep compared 100, 500, 1,000, and 5,000 fitting
particles and two learning rates at each count. All candidates were scored with
the Euler-20 model using 5,000 particles and 36 replicates. The best single
screening result was -3869.48 at 100 fitting particles and learning rate 0.10.
The best results at 500, 1,000, and 5,000 particles were -3891.60, -3940.57,
and -3970.87. The corresponding runs completed 86, 67, and 26 updates, versus
106 at 100 particles. Thus, under a fixed time budget, improved smoothing did
not compensate for the loss of optimizer updates on this start.

A focused 5,000-particle screen then compared learning rates 0.10, 0.15, and
0.20 and SA burn-ins from 5 to 30 updates. Learning rate 0.20 was best; its
burn-in-5 and burn-in-30 Euler-20 results were -3970.87 and -3971.13. The
burn-in-5 fit had the better block objective and starts decaying its learning
rate earlier, so that configuration was used for the final 100-start
experiment. The final experiment used 5,000 fitting particles, as requested,
even though the time-budget screen favored fewer particles.

The 800-second check showed genuine late deterioration. Euler-20 evaluations
with 5,000 particles and 36 replicates improved from -6206.05 initially to
-3919.90 at 715 seconds, then fell to -4069.20 at 809 seconds. The block guard
instead retained a stale iterate from about 549 seconds, whose final Euler-20
evaluation was -4072.32. The guard is therefore used only to reject unsafe
steps, not as a checkpoint-selection criterion. The final searches use a fixed
700-second early-stopping threshold (the completed update can carry elapsed
time slightly past that threshold), which remains below the requested
800-second cap.

In the final 100-start experiment, 90 fits reached the 800-second limit and 10
ended early after a non-finite path or score. All 100 selected parameter
vectors nevertheless had finite Euler-20 evaluations with 5,000 particles and
36 replicates. Their median was -4100.40, their 10th percentile was -5382.88,
and their best value was -3871.57. Twenty-four fits exceeded -4000, two
exceeded -3900, and none exceeded -3800. Comparable-effort IFAD-0.97 has median
-3745.77 and best value -3744.04. The requested expectation is therefore
supported strongly: IFAD outperforms this Ditlevsen-style extension on Dacca
under the common elapsed-time and Euler-20 evaluation criteria.

The final elapsed-time trace contains all 4,536 stored optimizer evaluations;
the earlier 100-second subsampling is not used. After 200 seconds, the block
surrogate and Euler-20 values have Spearman correlation 0.94 over the visited
points. Selecting each run's highest-surrogate checkpoint costs only 2.96
Euler log-likelihood units at the median relative to an oracle choice among
all of that run's checkpoints. Yet the oracle median is still -4051.94, and no
checkpoint exceeds -3800. A direct common-seed block-filter check also scores
the best comparable-effort IFAD-0.97 vector at -3829.71, versus -4032.73 for
the best selected Ditlevsen vector. The good Euler region is therefore not
excluded by the Gaussian surrogate. The main problem is reaching it from the
wide box with noisy score ascent and few updates. In the paired first 50
starts, the 100-particle fits completed a median 94 updates versus 47 at 5,000
particles and were better by 45.6 Euler log-likelihood units at the median.
The original paper instead used 100 particles, 80 SAEM iterations,
data-informed initialization, and exact model-specific M-steps for problems
with only three to six parameters.

The block method has no within-month particle genealogy because it integrates
out the 19 internal states. KSV is therefore unnecessary for this experiment.
It would become relevant if the intermediate states were explicitly retained
inside a conditional particle filter. The existing fixed-endpoint bridge code
is excluded from the headline fit; a true KSV comparison would require
a full forward CPF-BBS implementation and a separate invariance/mixing check.

## Reproduction

From this directory:

```bash
PYTHONPATH=.:../../pypomp \
  /home/kevin/anaconda3/envs/pypomp/bin/python -m pytest -q tests

PYTHONPATH=.:../../pypomp \
  /home/kevin/anaconda3/envs/pypomp/bin/python -m ditlevsen.validate

PYTHONPATH=.:../../pypomp \
  /home/kevin/anaconda3/envs/pypomp/bin/python -m ditlevsen.validation_plots
```

All final searches were generated by `make_starts`: every start is an
independent reproducible draw from the same physical-scale box used in
`../code/global_search/prep.py`. There is no published-point exception. Every
candidate was evaluated under the same 20-substep Euler POMP used by the DMOP
manuscript. Final likelihood evaluation used 5,000 particles and 36
replications, matching the manuscript; the lower-particle runs are validation
only.

## Layout

- `ditlevsen/validation_ho.py`: literal DS oscillator filter and analytic oracle.
- `ditlevsen/block_smc.py`: monthly block filter, path imputation, and score update.
- `ditlevsen/smc.py`: superseded local-transition filter and score update.
- `ditlevsen/smc_benchmark.py`: resumable global-box benchmark and independent
  Euler-20 likelihood evaluation.
- `ditlevsen/particle_sweep.py` and `ditlevsen/particle_sweep_plot.py`:
  resumable particle-count and learning-rate screen and its PNG summary.
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
- `results/block_smc_10/`: superseded ten-start monthly bootstrap pilot.
- `results/block_smc_guided_100/`: stopped 50-start guided-FFBSi diagnostic.
- `results/block_smc_guided_j5000_final_100/`: final 100-start run with
  5,000 fitting particles, an 800-second trajectory, and fixed 700-second
  output selection.
- `results/withdrawn_qml/`: rejected EKF/QML diagnostics, not final results.

Nothing here commits, pushes, or changes the Pypomp package.
