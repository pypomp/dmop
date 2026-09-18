# Corenflos differentiable particle filter

This directory contains a JAX implementation of the entropy-regularized
optimal-transport particle filter of Corenflos et al. (2021), specialized to
the Dacca benchmark. It follows the published FilterFlow implementation's
state scaling, epsilon annealing, averaged Sinkhorn iterations, gradient
stitching, and clipped transport adjoint. Pypomp itself is unchanged.

The fitting objective is the biased differentiable-particle-filter likelihood.
All reported comparisons are evaluated afterward with the common ordinary
Euler-20 particle-filter likelihood, as for the other methods in the DMOP
benchmark. Every stored optimizer update is evaluated for the elapsed-time
trace; the final estimate uses an independent 5,000-particle evaluation with
36 replicates. The selected estimate is the valid checkpoint with the largest
DPF likelihood before 700 seconds. Euler-20 values are not used to select it.

The transport geometry contains the full physical particle state: `S`, `I`,
`Mn`, `R1`, `R2`, and `R3`. The Pypomp `count` field is an invalid-state
diagnostic rather than a model coordinate and is not transported. A stochastic
filter call remains eligible while at least one particle is valid at every
observation; an all-invalid call is rejected and rolled back.

## Dacca results

The completed 100-start run is in
`results/corenflos_j100_eps025_final_100`. Each start was drawn from the same
global parameter box used for the other Dacca comparisons and had an
800-second fitting budget. Final estimates were scored with 5,000 particles
and 36 independent Euler-20 likelihood replicates. The optimization figure is
based on a separate 5,000-particle Euler-20 evaluation of every stored update
(27,163 evaluations, including the initial points).

| Method | Median log likelihood | Best | Runs at least -4300 |
|---|---:|---:|---:|
| Corenflos | -4554.67 | -3842.43 | 38/100 |
| Ditlevsen | -4100.34 | -3872.97 | 74/100 |
| IF2 | -3755.41 | -3747.82 | 100/100 |
| IFAD-0 | -3753.88 | -3748.61 | 100/100 |
| IFAD-0.97 | -3745.77 | -3744.04 | 99/100 |
| IFAD-1 | -3749.12 | -3745.08 | 96/100 |

The IFAD and IF2 rows use the manuscript's comparable-effort runs. Corenflos
occasionally reaches the Ditlevsen range, but its median is 454 log-likelihood
units lower and it succeeds much less often. Neither method approaches the
IFAD or IF2 distributions reliably. The audited figures are:

- `figures/likelihood_comparison_r20.png`
- `figures/parameter_comparison_r20.png`
- `figures/optimization_elapsed_r20.png`
- `figures/optimization_elapsed_full_r20.png` (the Corenflos median lies below
  the manuscript panel's -4300 cutoff)

## IF2-warm-start comparison

The follow-up comparison uses the exact 100 post-MIF checkpoints from the
comparable-effort IFAD-0.97 run. They are exported from the manuscript pickle
to `../ditlevsen/results/reference/ifad097_comparable_post_if2.npz`; the JSON
beside it records the source files and timings. IFAD-0.97 used 175 MIF updates
with 5,000 particles in 244.442 seconds, followed by at most 175 gradient
updates over another 614.686 seconds. The warm-started Corenflos and Ditlevsen
runs use the same update cap and remaining elapsed-time budget, and their trace
times include the 244.442-second IF2 offset. Thus all three branches end at the
same nominal 859.128-second point.

The Corenflos fit is written to
`results/if2warm_ifad097_budget_j100_final_100`; the combined comparison will
be written to `results/if2warm_ifad097_comparison`. The fit, baseline
evaluation, final evaluation, every-update evaluation, figures, and audit are
all resumable. The persistent scripts are:

- `scripts/run_if2warm_final100.sh`
- `scripts/evaluate_if2warm_starts.sh`
- `scripts/evaluate_if2warm_final100.sh`
- `scripts/evaluate_if2warm_pipeline.sh`

Run commands from this directory with:

```sh
export PYTHONPATH=.:../ditlevsen:../../pypomp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

Run the tests with `make test`. A resumable fit is, for example:

```sh
python -m corenflos.benchmark \
  --stages fit \
  --output results/corenflos_j100_eps025_final_100 \
  --starts 100 \
  --particles 100 \
  --maximum-elapsed-seconds 800 \
  --output-selection-seconds 700
```

The driver writes one atomic checkpoint per start. Running the same command
again skips completed fits. Use a user systemd service for a long job so it is
not tied to a VS Code terminal. Evaluation is a separate resumable stage:

```sh
python -m corenflos.benchmark \
  --stages trace-eval final-eval \
  --output results/corenflos_j100_eps025_final_100 \
  --starts 100 \
  --particles 100 \
  --trace-eval-particles 5000 \
  --trace-eval-replicates 1 \
  --eval-particles 5000 \
  --eval-replicates 36

python -m corenflos.plots --data results/corenflos_j100_eps025_final_100
```

References: [Corenflos et al. (2021)](https://proceedings.mlr.press/v139/corenflos21a.html)
and the [FilterFlow source](https://github.com/JTT94/filterflow).
