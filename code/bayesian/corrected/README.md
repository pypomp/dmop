# Corrected Bayesian experiment (in progress)

The archived NUTS calculation divided both log likelihood and log prior by
600. These new experiments remove that division and use current Pypomp,
without importing the archived diffPomp implementation. Historical arrays and
figures remain unchanged until replacement runs have passed their checks.

The primary comparison is NUTS versus particle random-walk Metropolis–Hastings
(PMMH). MALA and fixed-length HMC runs are retained as development trials.
See `results/gpu-comparison-v2/protocol.json` for the current GPU comparison.

## Target and model

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

The current batches are in `results/gpu-comparison-v2/nuts/chain{0,1,2,3}`
and `results/gpu-comparison-v2/pmrw/chain{0,1,2,3}`. They use the same starts,
proposal covariance, proper prior and 10,000-particle likelihood. NUTS uses
a 1,000-particle float32 force with deterministic GPU execution. Their
directories link the preparation, noise and cost records. Both batches have
saved finite initial checkpoints; convergence is still unverified.

The earlier 5,000-particle PMMH run continues under
`results/gpu-comparison-v1/pmrw-continued` for a particle-count check. Its
original `pmrw` directory is an immutable parent; the continuation retains
every parent draw and the same random state and frozen tuning. The earlier
GPU NUTS warmup was retired after validating the new deterministic run.

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
