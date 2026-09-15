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
36 replicates.

The transport geometry contains the full physical particle state: `S`, `I`,
`Mn`, `R1`, `R2`, and `R3`. The Pypomp `count` field is an invalid-state
diagnostic rather than a model coordinate and is not transported.

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
  --output results/final_100 \
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
  --output results/final_100 \
  --starts 100 \
  --particles 100 \
  --trace-eval-particles 5000 \
  --trace-eval-replicates 1 \
  --eval-particles 5000 \
  --eval-replicates 36

python -m corenflos.plots --data results/final_100
```

References: [Corenflos et al. (2021)](https://proceedings.mlr.press/v139/corenflos21a.html)
and the [FilterFlow source](https://github.com/JTT94/filterflow).
