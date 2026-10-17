# Corrected Bayesian experiment

The archived NUTS calculation divided both log likelihood and log prior by
600. These new experiments remove that division and use current Pypomp,
without importing the archived diffPomp implementation. Historical arrays and figures remain archived one directory above. The
manuscript now uses the common-target comparison described below, with the
remaining convergence limitations reported explicitly.

The primary comparison is NUTS versus particle random-walk Metropolis–Hastings
(PMMH). MALA and fixed-length HMC runs are retained as development trials.
The current NUTS configuration is in `results/gpu-forward-nuts-v2`; its
PMMH reference is complete in `results/gpu-comparison-v2`.

## Files to retain in Git

Each experiment keeps its launch script, protocol, configuration files and
initial inputs. For the identity-metric control, aborted fixed-step control
and wider-prior PMMH run, the batch directory also contains one `source/`
snapshot and a `reproducibility.json` manifest. The manifest records source,
configuration and input hashes; input paths are relative to the dmop root.
The four per-chain source copies were checked against this shared snapshot.

Live checkpoints, status/metrics files, duplicate per-chain sources and
initial arrays, and OneDrive `safeBackup` files are ignored. They remain on
disk for running jobs and continuation. Existing tracked results remain
tracked. Frozen review reports and manuscript figures remain versioned.
At the end of a run, archive the selected final checkpoint explicitly
(`git add -f` for that checkpoint only), together with its review report.
Do not add sync backups or intermediate checkpoints.

For a new run, retain its configuration and launch inputs and save one source
snapshot with hashes before changing the implementation. Run the archived
launcher against that snapshot to reproduce the experiment. Configuration
files record the original machine's absolute paths; use the manifest's
repository-relative input paths on another machine. To continue an archived
chain after cloning, restore its local `source/` from the batch snapshot and
its `initial.npy` and `initial_chol.npy` from the manifest's input paths.
The checkpoint selected for continuation must also be available.

## Manuscript figures and diagnostic table

Run from the dmop root with the report dependencies installed:

```sh
python code/bayesian/corrected/paper_figures.py
```

This reads frozen reports; it does not run a sampler. The primary NUTS
report is `results/gpu-forward-nuts-v2/review-nuts-2000`. The PMMH reports
are `results/gpu-comparison-v2/review-pmrw-2000` and `review-pmrw-30000`.
The script checks matching models/targets and the PMMH prefix before plotting.

- `imgs/pmcmc/common_target_trend.pdf`: main Figure 6, four chains per method,
  first 2,000 retained iterations after 500 warmup updates. Trace, density,
  and autocorrelation panels use shared linear axes.
- `results/paper/diagnostics.tex`: main-text and SI table, all 18 coordinates
  at that same 2,000-iteration window. Bold cells select R-hat closest to 1,
  higher bulk/tail ESS or lower mean MCSE; displayed ties are both bold.
- `imgs/pmcmc/common_target_reference.pdf`: SI marginal comparison with the
  30,000-iteration PMMH reference. Pool chains separately by method. For this
  display alone, retain PMMH iterations 25, 50, ..., 30000 (1,200 per chain).
  NUTS has 2,000 retained draws per chain. PMMH is dark orange/dashed;
  NUTS is blue/filled. Each coordinate uses a common absolute Gaussian KDE
  bandwidth, selected by Scott's rule from the thinned PMMH sample.
  Thinning does not assert independence. Numerical diagnostics use all draws.
- `imgs/pmcmc/common_target_traces.pdf` and `common_target_marginals.pdf`:
  SI NUTS traces and prior overlays for all 18 coordinates, shown by chain.

PNG previews accompany the vector PDFs. `results/paper/manifest.json`
records input hashes, windows, plotting conventions and bold-cell choices.
All four figures and their placement in the compiled documents were visually
checked. The figure script, not an external notebook, is their source.

The primary NUTS run is complete at 500 warmup + 2,000 retained iterations
per chain. Trend R-hat is 1.014 and bulk ESS is 333. Across coordinates,
maximum R-hat is 1.059, minimum bulk ESS is 60 and minimum tail ESS is 116.
Two retained transitions diverged; almost all trajectories reached depth 5.
These results show improved exploration per iteration, not established joint
convergence or a computing-time advantage. The final report preserves all
saved draws and verifies the earlier 1,000-draw review as an exact prefix.

## Original-settings control and historical comparison

The fixed-step original-settings NUTS control was stopped at the user's
request after 250 saved iterations per chain, with frequent divergences.
`results/original-settings-nuts-v1/termination.json` preserves process identity
and checkpoint hashes. This aborted run is not used as a posterior sample.

Its replacement in `results/identity-adapt-nuts-v1/nuts` uses the same fixed
identity metric in original estimation coordinates, but adapts the scalar
step size during 500 warmup iterations, starting at .002 and targeting .65
trajectory acceptance. It then retains 1,000 iterations with frozen tuning.
The four starts, common prior, untempered 10,000-particle target and
1,000-particle DMOP force are unchanged. The metric does not adapt.

Compare with the completed diagonal PMMH control in
`results/historical-proposal-control-v1/pmrw` (fixed standardized SD .02,
no adaptation, 1,500 iterations). Use iterations 501--1500 for both methods,
with full traces showing the first 500 separately. The first 500 PMMH draws
are discarded for this comparison, not an adaptive warmup. The additional
SI figure is pending completion of the replacement NUTS control. This is a
comparison of identity-metric NUTS with adapted steps versus fixed diagonal
PMMH; do not describe it as a replication of the archived NUTS settings.
Automatic polling is paused; do not restart or extend runs without a request.

Fasiolo et al. (2016) proposal provenance remains unresolved. The paper and
its eight-page supplement specify POMP PMMH but not the proposal covariance.
The October 17 search checked the author's public repository list, the
`synlik_dev` master and `synlik_CRAN` trees, `synlik_release`, `LFexamples`,
the authors' publication pages, and GitHub repository searches for the paper.
No original Dhaka analysis script was located. `LFexamples` links to the
library's Dhaka simulator, not the paper's PMMH analysis. Do not infer a
specific proposal from library capabilities or label our diagonal control
as Fasiolo's implementation.
Sources: [paper](https://doi.org/10.1214/15-STS534),
[supplement](https://doi.org/10.1214/15-STS534SUPP),
[author repositories](https://github.com/mfasiolo?tab=repositories),
[example list](https://github.com/mfasiolo/LFexamples).

## Target and model

For sensitivity checks, first freeze each setting with `report.py`, then use
`sensitivity.py --reference REPORT1 --alternative REPORT2 --kind prior`
`--output NEW_DIRECTORY` (or `--kind particles`). It requires the same
method and model, permits only the requested target-setting difference,
and shows individual chains. Its table reports marginal intervals, standard
deviations, mean shifts and their Monte Carlo errors. Passing convergence
checks is required before interpreting these comparisons; no sensitivity
claim is made automatically.

The target is the untempered posterior for the Pypomp Gaussian-noise Dhaka
Euler model, with 20 steps per monthly observation and the library's fixed
initial proportions. There are 18 estimated parameters. The obsolete constant
omega coordinate is absent; rho=0, c=1, alpha=1, delta=.02 and initial
proportions are fixed. Pypomp normalizes those initial proportions and
interpolates the covariates; these conventions differ slightly from the old
notebook.

The experiment uses Pypomp's simulator and particle filters. The sole
measurement change removes its additive 1e-18 density floor. The observation
model is Gaussian with mean cumulative monthly deaths and SD tau times that
mean. Euler paths that have crossed a negative state are sent to an absorbing
failure state, whose observation is outside the real-valued observed series.
Their likelihood is zero. This defines a normalized observation model, unlike
adding a constant to a density over the whole real line. Pilot checks compare
the modified likelihood to the library's original objective.

A common proper reference prior replaces the chain-specific empirical KDEs.
The 18 estimation coordinates have independent Normal priors, with mean at
the midpoint of the old search ranges and SD half their width. These ranges
and the coordinate transformations are explicit in `model.py`. They are a
reference choice, not claimed scientific prior knowledge. The completed
doubled-SD sensitivity run is described below. IF2 or optimization
may supply starting points and proposal scales; neither changes the prior.

Removing the empirical KDE does not justify an unrestricted flat prior.
In the library model, environmental immigration is
`exp(sum_k seas_k(t) * omegas_k)`. The supplied seasonal basis is nonnegative
and sums to one, so subtracting a common constant c from all six coefficients
multiplies immigration by `exp(-c)`. This approaches the model with no
environmental immigration. If its limiting data likelihood is positive,
a flat prior on that common level has a nonintegrable lower tail. We have not
evaluated that limiting likelihood here. The observed long tails in individual
coefficients also require checking; this algebra alone does not diagnose their
mixing or establish prior sensitivity. The doubled-SD experiment below
examines this dependence while retaining a proper prior.

## Sampler and checks

The extended target is proportional to `Lhat(theta,u) prior(theta) m(u)`,
where u is an explicit particle-filter PRNG state. The unscaled bootstrap
particle-filter estimate enters every Metropolis ratio. MOP supplies only a
proposal force. Conditional MALA, HMC and NUTS alternate parameter updates
at fixed u with an independent proposal from m, accepted with the likelihood
ratio. Joint particle MALA and random-walk MH instead propose parameters and
u together. Retaining u on rejection is essential. The force is deterministic
conditional on u.
A fixed-length reversible leapfrog proposal with an endpoint energy correction
is valid even if this force is not an exact gradient. We do not infer validity
of an adaptive NUTS stopping rule from that argument.

Before full runs: test reversibility and known-target moments, check agreement
of MOP's forward value with the independent PF API at identical keys, estimate
PF variability and timing, and check the effect of removing the density floor.
Adaptation is restricted to warmup. Save every retained draw and rejection,
seeds, settings, source hashes, runtime and acceptance/energy diagnostics.

Four independent chains must agree (rank split R-hat <=1.01, bulk and tail
ESS >=400 for every estimated parameter) and traces must be visually checked.
Prior sensitivity and doubled-particle checks follow. If these checks fail,
retain the failed runs and tune/extend sampling. Aesthetic similarity to the
old plots is not a convergence criterion. Keep the existing figure layout
where it remains informative; revise empirical claims only as needed.

## Running and reviewing

- `preflight.py`: PF/MOP agreement, observation-floor effect, noise and timing.
- `noise_check.py`: batched independent likelihood estimates at saved sampling
  states, including a diagnostic for sticking caused by the retained filter
  estimate. It compares particle counts without changing an ongoing chain.
- `coupling.py`: paired likelihood estimates with optional endpoint sorting.
- `kernels.py`: corrected proposal kernels; see [METHOD.md](METHOD.md).
- `nuts.py`: BlackJAX NUTS with deterministic MOP forces, unscaled PF
  energies, and an explicit MH update of particle randomness.
- `run.py`: one chain, complete checkpoint and warmup history, frozen source.
- `batch.py`: independent NUTS or PMMH chains executed together, with separate
  seeds, tuning histories and per-chain files compatible with `report.py`.
  Supply equally sized `--initials`, `--metrics` and `--seeds` lists. All
  likelihoods use float64. Optional `--force-precision float32` affects only
  proposal gradients. Per-chain timing fields contain the entire batch's
  wall time; do not sum them across chains to estimate elapsed time.
  A scalar continuation can change float32 force rounding, so `extend.py`
  rejects these batched checkpoints. Use `batch_extend.py` to preserve the
  original batch size, lane order and cached random states.
- `batch_extend.py`: continue a completed batch with `--parent BATCH_DIR
  --output NEW_DIR --draws COUNT`. Explicit `--recover-interrupted` instead
  requires a batch-level `interruption.json` recording a verified terminal
  process and retains the original run length. It handles unequal saved
  iteration counts after a partial write. Every output chain includes its
  parent trace; never pool parent and continuation as separate chains.
  CPU replay is bitwise identical in the recorded test. With the default GPU
  execution settings, repeated float32 force evaluations vary slightly;
  GPU replay preserves random states but is not bitwise identical. The
  optimized deterministic XLA settings also give bitwise replay on the GPU
  in the four-chain interrupted-warmup test. See
  `results/batch-continuation-check-v1/optimized-verification.json` and
  `results/gpu-precision-check-v1/optimized-deterministic-batch4`.
  Both continuation drivers require the parent's recorded XLA flags.
- `extend.py`: exact continuation into a new directory, preserving the parent,
  tuning and random state. `--recover-interrupted` requires a recorded terminal
  interruption and resumes the original run length and warmup schedule.
- `metric.py`: proposal scaling from replicated scores; affects efficiency
  only, and is not a posterior covariance estimate.
- `device_check.py`: GPU precision and batch-throughput measurements at fixed
  saved points; separate from posterior sampling. Default model precision is
  float64; an explicit `JAX_ENABLE_X64=false` selects float32 before data import.
  `--force-precision float32 --precision float64` instead retains the full
  precision likelihood and measures a cheaper proposal force. This option
  does not temper the target or change its particle count.
- `force_check.py`: force variability and serial cost at the same fixed point
  and CPU affinity, for selecting a gradient particle count.
- `conditional_force_check.py`: local fixed-seed finite differences and the
  effect of crossing resampling discontinuities.
- `report.py`: all-coordinate diagnostics and the original four-panel trend
  layout, plus full trace/marginal grids with common-prior overlays.
  Use `--max-retained N` to review the first N post-warmup draws per chain
  while sampling continues. Each report saves the raw arrays it used in
  `review_snapshot.npz`, as well as transformed posterior draws.
- `compare.py`: compare frozen NUTS/PMMH reports on the same target, showing
  all-coordinate marginal agreement and differences relative to Monte Carlo
  error. Unconverged reports remain explicitly marked as development samples.
  Optional `--trend-panels` adds a two-row comparison of trend traces,
  densities and autocorrelations, with shared linear axes. This requires equal
  chain counts and retained lengths; use `report.py --max-retained N` to freeze
  the same prefix for each method. Warmup is excluded. Matching chain colors
  identify chains within each method, not paired random draws. Equal iterations
  do not imply equal computing time. The default marginal comparison also
  accepts unequal lengths, for comparison with a longer reference run.
- `results/`: all pilots, including unsuccessful settings. No pilot is a
  final posterior sample. The historical files one level up stay unchanged.

Example from dmop, with Pypomp/JAX installed:

```sh
JAX_ENABLE_X64=true python code/bayesian/corrected/run.py \
  --method nuts --particles 5000 --force-particles 1000 --seed 2026101600 \
  --output code/bayesian/corrected/results/new-chain
```

Use distinct seeds and disjoint output directories for independent chains.
`report.py` additionally requires ArviZ and Matplotlib. It checks target
settings and chain seeds before pooling. Never run with an existing output
folder: overwrite protection preserves failed tuning as well as successful
runs. Checkpoints retain the random state needed to continue the chain.

The current comparison is MOP-driven NUTS against ordinary particle random-walk
MH. Joint particle MALA and fixed-length HMC remain development trials. The joint MALA
proposal refreshes parameters and randomness together, with its reverse
proposal density in the acceptance ratio. See METHOD.md for each kernel.

The current batches each contain `chain0` through `chain3`:

| Purpose | Batch directory under `results/` |
|---|---|
| Primary NUTS | `gpu-forward-nuts-v2/nuts` |
| Primary PMMH | `gpu-comparison-v2/pmrw-long` |
| PMMH with 5,000 particles | `gpu-comparison-v1/pmrw-long` |
| PMMH with doubled prior SDs | `prior-sensitivity-gpu-v1/pmrw-60000` |
| Fixed historical-proposal control | `historical-proposal-control-v1/pmrw` |

Primary NUTS and PMMH use the same starts, proper prior and 10,000-particle
likelihood. NUTS uses
a 1,000-particle float32 force with forward differentiation and deterministic
GPU execution. `force_modes.py` differentiates the scalar objective supplied
by the unchanged model. `--force-mode` is saved with every new run, and
continuation restores it from the parent's frozen source. The NUTS proposal
covariance and initial step size were revised after the paired screen in
`results/nuts-metric-check-v1`. Its local movement improvement is not an
ESS/sec or convergence result. Earlier warmups are preserved in
`gpu-forward-nuts-v1/nuts` and `gpu-comparison-v2/nuts`; their termination
records identify the retired processes. Include the covariance pilot,
tuning screens and retired warmups in setup costs. Convergence is still
unverified.

Primary PMMH completed 30,000 retained draws per chain. Its report in
`gpu-comparison-v2/review-pmrw-30000` shows improved agreement relative to
10,000 draws, with residual seasonal dependence and unequal tail exploration.
All 18 traces and marginal plots have been inspected. The largest rank
R-hat is 1.014; the smallest bulk and tail ESS are 202 and 152. These
results do not establish joint convergence. The completed extension in
`gpu-comparison-v2/pmrw-long` preserves the original target, random state
and frozen tuning, and includes the entire original `pmrw` trace.
Report the latest continuation without pooling overlapping files.
`review-pmrw-2000` freezes its first 2,000 retained draws for the main
equal-iteration NUTS comparison; this does not equalize computing time.

The 5,000-particle PMMH run also completed 30,000 retained draws per chain
in `results/gpu-comparison-v1/pmrw-long`. Its report is
`gpu-comparison-v1/review-pmrw-30000`; comparison with the primary run is
in `gpu-comparison-v2/particle-comparison-30000`. All 18 displayed marginal
distributions broadly overlap, with residual chain differences and tail
uncertainty. This is a descriptive particle-count check, not evidence of
joint convergence or general insensitivity to particle count. The extension
retains every draw from `pmrw-continued`, including the original `pmrw`
prefix. Use only the latest continuation when reporting these chains.

The wider-prior experiment completed 60,000 retained draws per chain in
`results/prior-sensitivity-gpu-v1/pmrw-60000`. It doubles all prior standard
deviations while keeping the centers and likelihood fixed. The frozen report
is `prior-sensitivity-gpu-v1/review-pmrw-60000`; `comparison-60000` compares
it with the primary 30,000-draw reference. All 18 traces and marginals have
been inspected. The sampled trend SD changes from 0.1121 to 0.1151, whereas
the SDs of environmental coefficients 1, 5 and 6 increase by factors of
1.81, 1.73 and 2.49. Tail occupancy remains unequal across chains; the
largest R-hat is 1.074 and smallest bulk ESS is 56.6, both for coefficient 6.
These finite-run results support replacing the empirical KDE with a common
proper prior, but do not establish prior insensitivity. No further
sensitivity continuation is scheduled. See that directory's README and
numerical review for the retained limitations and exact continuation checks.

The historical-proposal control uses the current primary starts and target
with the original fixed diagonal proposal for 1,500 iterations. Its purpose
is to investigate the drift in the original SI traces. It changes proposal
choice and adaptation, while retaining the common prior and untempered
likelihood. The original traces and available sampling source are archived
in [historical-mh](../historical-mh/README.md); that directory also documents
the missing information that prevents an exact fresh replay. The finite-length
comparison is separate from the longer posterior-reference runs.
The control has completed: acceptance is 9.6--13.6%, with substantial chain
separation, but without the original trend excursion to -1.22. Its
[review and figures](results/historical-proposal-control-v1/README.md) compare
all three runs at 1,500 iterations and distinguish their targets.

The CPU runs in `results/candidate-nuts-4chain-v1` and
`results/sampling-recovery-v1` were retired after validating the GPU starts;
their `termination.json` files record the verified processes and checkpoints.
These remain development results. Each PMMH recovery includes its complete
parent trace; do not count the parent as another chain or pool these old
chains with the new GPU runs.

Generate each method report with `report.py`, then run:

```sh
python code/bayesian/corrected/compare.py --nuts PATH_TO_NUTS_REPORT \
  --pmmh PATH_TO_PMMH_REPORT --output PATH_TO_COMPARISON
```

The comparison uses the exact arrays frozen with each report. Different
warmup lengths are handled within each method report. Its MCSE ratios are
descriptive and do not establish equivalence or convergence.
