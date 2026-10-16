# Corrected Bayesian experiment (in progress)

The archived NUTS calculation divided both log likelihood and log prior by
600. These new experiments remove that division and use current Pypomp,
without importing the archived diffPomp implementation. Historical arrays and
figures remain unchanged until replacement runs have passed their checks.

The primary comparison is NUTS versus particle random-walk Metropolis–Hastings
(PMMH). MALA and fixed-length HMC runs are retained as development trials.
The current NUTS configuration is in `results/gpu-forward-nuts-v2`; its
PMMH counterpart continues in `results/gpu-comparison-v2`.

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
reference choice, not claimed scientific prior knowledge. A doubled-SD
sensitivity run will assess the dependence on that choice. IF2 or optimization
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
mixing or establish prior sensitivity. Retain the proper prior and compare
with doubled prior SDs after both runs have converged.

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
- `compare.py`: compare frozen NUTS/PMMH reports on the same target, showing
  all-coordinate marginal agreement and differences relative to Monte Carlo
  error. Unconverged reports remain explicitly marked as development samples.
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
| PMMH with doubled prior SDs | `prior-sensitivity-gpu-v1/pmrw-long` |
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

Primary PMMH completed 10,000 retained draws per chain; its report in
`gpu-comparison-v2/review-pmrw-10000` still shows disagreement in seasonal
coefficients and incomplete tail exploration. The same chains continue to
30,000 retained draws in `gpu-comparison-v2/pmrw-long`, preserving the target,
random state and frozen tuning. The extension includes the entire original
`pmrw` trace; report the latest continuation without pooling overlapping files.

The earlier 5,000-particle PMMH run continues under
`results/gpu-comparison-v1/pmrw-long` for a particle-count check. The completed
10,000-draw report is in `gpu-comparison-v1/review-pmrw-10000`; it did not
pass convergence checks. The extension targets 30,000 retained draws per
chain. It retains every draw from `pmrw-continued`, which itself includes
the original `pmrw` prefix, with the same random state and frozen tuning.
Use only the latest continuation when reporting these chains; never pool
overlapping parent files.

The wider-prior comparison is running in `results/prior-sensitivity-gpu-v1`.
It doubles the normal prior's standard deviations while keeping its centers
and the likelihood fixed. It started after the initial 10,000-draw
5,000-particle PMMH continuation finished. See its protocol, launch record
and README for settings and review steps. Its completed 10,000-draw report
shows substantial immigration-coefficient chain separation, although the
trend distribution resembles the original-prior result. The same chains
continue to 30,000 retained draws in `pmrw-long`, with unchanged tuning.
`comparison-10000` retains the provisional comparison of both prior settings
at 10,000 draws. Prior sensitivity remains unresolved.

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
