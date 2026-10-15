# Corrected Bayesian experiment (in progress)

The archived NUTS calculation divided both log likelihood and log prior by
600. These new experiments remove that division and use current Pypomp,
without importing the archived diffPomp implementation. Historical arrays and
figures remain unchanged until replacement runs have passed their checks.

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
- `coupling.py`: paired likelihood estimates with optional endpoint sorting.
- `kernels.py`: corrected proposal kernels; see [METHOD.md](METHOD.md).
- `nuts.py`: BlackJAX NUTS with deterministic MOP forces, unscaled PF
  energies, and an explicit MH update of particle randomness.
- `run.py`: one chain, complete checkpoint and warmup history, frozen source.
- `extend.py`: exact continuation from a completed checkpoint into a new
  directory, preserving the parent, frozen tuning and random state.
- `metric.py`: proposal scaling from replicated scores; affects efficiency
  only, and is not a posterior covariance estimate.
- `report.py`: all-coordinate diagnostics and the original four-panel trend
  layout, plus full trace/marginal grids with common-prior overlays.
- `results/`: all pilots, including unsuccessful settings. No pilot is a
  final posterior sample. The historical files one level up stay unchanged.

Example from dmop, with Pypomp/JAX installed:

```sh
JAX_ENABLE_X64=true python code/bayesian/corrected/run.py \
  --method pmmala --particles 5000 --seed 2026101600 \
  --output code/bayesian/corrected/results/new-chain
```

Use distinct seeds and disjoint output directories for independent chains.
`report.py` additionally requires ArviZ and Matplotlib. It checks target
settings and chain seeds before pooling. Never run with an existing output
folder: overwrite protection preserves failed tuning as well as successful
runs. Checkpoints retain the random state needed to continue the chain.

The current candidates are NUTS, joint particle MALA and fixed-length HMC powered
by MOP, together with ordinary particle random-walk MH. The joint MALA
proposal refreshes parameters and randomness together, with its reverse
proposal density in the acceptance ratio. See METHOD.md for each kernel.
