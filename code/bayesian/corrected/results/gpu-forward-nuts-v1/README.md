# NUTS with forward differentiation

This supplies the NUTS chains for the comparison with PMMH in
[`../gpu-comparison-v2/pmrw`](../gpu-comparison-v2/pmrw). Both use the same
untempered 10,000-particle likelihood, common normal prior, four starts and
fixed proposal covariance. NUTS uses a 1,000-particle float32 proposal force;
the likelihood and acceptance calculations remain float64.

The optional `--force-mode forward` differentiates the scalar objective from
the existing model using `jax.jacfwd`. It reduces the cost of constructing
the proposal without modifying model.py or fitting a prior. Numerical and
full-kernel checks, including exact CPU/GPU interrupted-run recovery, are in
[`../forward-force-check-v1`](../forward-force-check-v1).

`protocol.json` records the settings and frozen input/source hashes.
`launch.py` refuses to overwrite a run. `startup-verification.json` verifies
finite states and matching target, prior, starts and covariance in all four
chains. Configurations, frozen source and checkpoints are in `nuts/chain0`
through `nuts/chain3`. Use `batch_extend.py` with the recorded XLA flags for
continuation; the parent configuration determines the force mode.

The older reverse-mode warmup is preserved in
`../gpu-comparison-v2/nuts`, with a verified termination record. Count its
cost, together with all preparation, when assessing computational performance.
The first new update took 88.57 seconds including compilation while both
NUTS jobs shared GPU0. This is not a controlled speedup measurement.

These are ongoing experiments. Finite checkpoints and successful recovery
checks do not establish convergence. Review movement and depth limits after
50 warmup updates, and assess all-parameter convergence, prior sensitivity
and particle-count agreement before replacing the manuscript figures.
